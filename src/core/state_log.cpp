#include <madrona/state_log.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/sync.hpp>

#include <filesystem>
#include <fstream>
#include <thread>

namespace madrona {

namespace {

struct LogReaderFile {
    std::fstream hdl;
    uint32_t numBytesPerEntry;
};

struct LogWriterBuffer {
    void *ptr;
    uint64_t numSavedBytes;
    uint64_t numAllocatedBytes;
};

}

struct StateLogReader::Impl {
    std::fstream indexFile;
    HeapArray<LogReaderFile> logFiles;

    static inline Impl * init(const Config &cfg, const char *dir);
};

struct StateLogWriter::Impl {
    std::filesystem::path logDir;

    LogWriterBuffer indexBuffers[2];
    HeapArray<LogWriterBuffer> logBuffers[2];
    HeapArray<uint32_t> numBytesPerLogEntry;

    int32_t curSaveBuffer;
    int32_t numBufferedSteps[2];
    int32_t numStepsBeforeFlush;
    uint32_t numWorlds;

    std::fstream indexFile;
    HeapArray<std::fstream> logFiles;

    AtomicU32 diskWriteWakeup;
    std::thread diskWriteThread;

    static inline Impl * init(const Config &cfg, const char *dir);

    inline void addStepLogs(Span<LogEntries> step_data);
    inline void saveBuffersToDisk();

    inline void diskWriterLoop();
    inline void writeLogs(uint32_t write_buffer_idx);
};

static inline void writeU32(std::fstream &f, uint32_t v)
{
    f.write((char *)&v, sizeof(uint32_t));
}

StateLogReader::Impl * StateLogReader::Impl::init(const Config &cfg,
                                                  const char *dir)
{
    std::filesystem::path log_dir(dir);
    if (!std::filesystem::exists(log_dir)) {
        FATAL("State log directory %s not found", dir);
    }

    HeapArray<LogReaderFile> files(cfg.numBytesPerLogType.size());
    for (CountT i = 0; i < files.size(); i++) {
        std::fstream log_file(log_dir / std::filesystem::path(
            std::to_string(i) + ".log"),
            std::ios::binary | std::ios::out);

        new (&files[i]) LogReaderFile {
            .hdl = std::move(log_file),
            .numBytesPerEntry = cfg.numBytesPerLogType[i],
        };
    }

    std::fstream index_file(log_dir / std::filesystem::path("index"),
        std::ios::binary | std::ios::out);

    return new Impl {
        .indexFile = std::move(index_file),
        .logFiles = std::move(files),
    };
}

StateLogWriter::Impl * StateLogWriter::Impl::init(const Config &cfg,
                                                  const char *dir)
{
    std::filesystem::path log_dir(dir);
    if (std::filesystem::exists(log_dir)) {
        FATAL("State log directory %s already exists", dir);
    }

    bool created_log_dir = std::filesystem::create_directory(log_dir);
    if (!created_log_dir) {
        FATAL("Failed to create directory %s", dir);
    }

    std::fstream index_file(log_dir / std::filesystem::path("index"),
        std::ios::binary | std::ios::out);
    writeU32(index_file, cfg.numWorlds);

    CountT num_log_types = cfg.numBytesPerLogType.size();

    HeapArray<std::fstream> files(num_log_types);
    HeapArray<LogWriterBuffer> log_buffers[2] {
        HeapArray<LogWriterBuffer>(num_log_types),
        HeapArray<LogWriterBuffer>(num_log_types),
    };

    HeapArray<uint32_t> num_bytes_per_log_entry(num_log_types);

    for (CountT i = 0; i < files.size(); i++) {
        log_buffers[0][i] = { nullptr, 0, 0 };
        log_buffers[1][i] = { nullptr, 0, 0 };

        num_bytes_per_log_entry[i] = cfg.numBytesPerLogType[i];

        new (&files[i]) std::fstream(log_dir / std::filesystem::path(
            std::to_string(i) + ".log"),
            std::ios::binary | std::ios::out);

    }

    return new Impl {
        .logDir = std::move(log_dir),
        .indexBuffers = { { nullptr, 0, 0 }, { nullptr, 0, 0 } },
        .logBuffers = { std::move(log_buffers[0]), std::move(log_buffers[1]) },
        .numBytesPerLogEntry = std::move(num_bytes_per_log_entry),
        .curSaveBuffer = 0,
        .numBufferedSteps = { 0, 0 },
        .numStepsBeforeFlush = (int32_t)cfg.numBufferedSteps,
        .numWorlds = cfg.numWorlds,
        .indexFile = std::move(index_file),
        .logFiles = std::move(files),
        .diskWriteWakeup = 0,
        .diskWriteThread = std::thread(),
    };
}

static void saveRawToBuffer(LogWriterBuffer &buffer,
                            void *data,
                            uint64_t num_data_bytes)
{
    uint64_t new_num_saved_bytes = num_data_bytes + buffer.numSavedBytes;
    if (new_num_saved_bytes > buffer.numAllocatedBytes) {
        buffer.numAllocatedBytes =
            std::max(buffer.numAllocatedBytes * 2, new_num_saved_bytes);
        buffer.ptr = realloc(buffer.ptr, buffer.numAllocatedBytes);
    }

    memcpy((char *)buffer.ptr + buffer.numSavedBytes,
           data, num_data_bytes);
    buffer.numSavedBytes = new_num_saved_bytes;
}

template <typename T>
static void saveNToBuffer(LogWriterBuffer &buffer,
                          T *data,
                          CountT n)
{
    uint64_t num_data_bytes = sizeof(T) * (uint64_t)n;
    saveRawToBuffer(buffer, data, num_data_bytes);
}

template <typename T>
static void saveToBuffer(LogWriterBuffer &buffer,
                         T v)
{
    saveNToBuffer<T>(buffer, &v, 1);
}

void StateLogWriter::Impl::addStepLogs(Span<LogEntries> step_data)
{
    int32_t cur_buffer_idx = curSaveBuffer;
    int32_t cur_num_buffered_steps = numBufferedSteps[cur_buffer_idx]++;

    LogWriterBuffer &cur_idx_buffer = indexBuffers[cur_buffer_idx];
    auto &cur_log_buffers = logBuffers[cur_buffer_idx];

    for (CountT log_type_idx = 0; log_type_idx < cur_log_buffers.size();
         log_type_idx++) {
        const LogEntries &new_entries = step_data[log_type_idx];
        LogWriterBuffer &log_buffer = cur_log_buffers[log_type_idx];

        uint32_t num_type_total_entries = new_entries.numTotalEntries;
        saveToBuffer<uint32_t>(cur_idx_buffer, num_type_total_entries);

        if (num_type_total_entries == 0) {
            continue;
        }

        saveNToBuffer<uint32_t>(
            cur_idx_buffer, new_entries.worldOffsets, numWorlds);
        saveNToBuffer<uint32_t>(
            cur_idx_buffer, new_entries.worldCounts, numWorlds);

        uint64_t num_bytes_per_entry = (uint64_t)numBytesPerLogEntry[log_type_idx];

        saveRawToBuffer(log_buffer, new_entries.data, 
            num_bytes_per_entry * (uint64_t)num_type_total_entries);
    }

    if (cur_num_buffered_steps == numStepsBeforeFlush - 1) {
        saveBuffersToDisk();
    }
}

void StateLogWriter::Impl::saveBuffersToDisk()
{
    diskWriteWakeup.wait<sync::acquire>(1);

    curSaveBuffer = curSaveBuffer ^ 1;

    diskWriteWakeup.store_release(1);
    diskWriteWakeup.notify_one();
    
    indexBuffers[curSaveBuffer].numSavedBytes = 0;
    auto &cur_log_buffers = logBuffers[curSaveBuffer];

    for (CountT log_type_idx = 0; log_type_idx < logFiles.size();
         log_type_idx++) {
        LogWriterBuffer &log_buffer = cur_log_buffers[log_type_idx];
        log_buffer.numSavedBytes = 0;
    }
    numBufferedSteps[curSaveBuffer] = 0;
}

void StateLogWriter::Impl::diskWriterLoop()
{
    while (true) {
        diskWriteWakeup.wait<sync::relaxed>(0);
        uint32_t ctrl = diskWriteWakeup.load_acquire();

        if (ctrl == 0) {
            continue;
        } else if (ctrl == 0xFFFF'FFFF) {
            break;
        }

        uint32_t cur_buffer = curSaveBuffer ^ 1;

        writeLogs(cur_buffer);

        diskWriteWakeup.store_release(0);
        diskWriteWakeup.notify_one();
    }
}

void StateLogWriter::Impl::writeLogs(uint32_t write_buffer_idx)
{
    LogWriterBuffer &cur_idx_buffer = indexBuffers[write_buffer_idx];
    auto &cur_log_buffers = logBuffers[write_buffer_idx];

    indexFile.write((char *)cur_idx_buffer.ptr, cur_idx_buffer.numSavedBytes);

    for (CountT log_type_idx = 0; log_type_idx < logFiles.size();
         log_type_idx++) {
        std::fstream &log_file = logFiles[log_type_idx];
        LogWriterBuffer &log_buffer = cur_log_buffers[log_type_idx];

        log_file.write((char *)log_buffer.ptr, log_buffer.numSavedBytes);
    }
}

StateLogReader::StateLogReader(const Config &cfg,
                               const char *dir)
    : impl_(Impl::init(cfg, dir))
{}

StateLogReader::StateLogReader(StateLogReader &&) = default;
StateLogReader::~StateLogReader() = default;
StateLogReader & StateLogReader::operator=(StateLogReader &&) = default;

StateLogWriter::StateLogWriter(const Config &cfg,
                               const char *dir)
    : impl_(Impl::init(cfg, dir))
{
    impl_->diskWriteThread = std::thread([this]() {
        impl_->diskWriterLoop();
    });
}

StateLogWriter::StateLogWriter(StateLogWriter &&) = default;
StateLogWriter::~StateLogWriter()
{
    impl_->diskWriteWakeup.store_release(0xFFFF'FFFF);
    impl_->diskWriteWakeup.notify_one();
    impl_->diskWriteThread.join();
}

StateLogWriter & StateLogWriter::operator=(StateLogWriter &&) = default;

void StateLogWriter::addStepLogs(Span<LogEntries> step_data)
{
    impl_->addStepLogs(step_data);
}

}
