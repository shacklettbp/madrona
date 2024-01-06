#include <madrona/state_log.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/sync.hpp>

#include <filesystem>
#include <fstream>

namespace madrona {

namespace {

struct LogReaderFile {
    std::fstream hdl;
    uint32_t numBytesPerEntry;
};

struct LogWriteBuffer {
    void *ptr[2];
    uint64_t numSavedBytes[2];
    uint64_t numAllocatedBytes[2];
};

struct LogWriterFile {
    std::fstream hdl;
    LogWriteBuffer buffer;
    uint32_t numBytesPerEntry;
};

}

struct StateLogReader::Impl {
    std::fstream indexFile;
    HeapArray<LogReaderFile> logFiles;

    inline Impl * init(const Config &cfg, const char *dir);
};

struct StateLogWriter::Impl {
    std::filesystem::path logDir;
    std::fstream indexFile;
    LogWriteBuffer indexBuffer;
    HeapArray<LogWriterFile> logFiles;
    
    int32_t curSaveBuffer;
    int32_t numBufferedSteps[2];
    AtomicU32 diskWriteWakeup;
    std::thread diskWriteThread;

    inline Impl * init(const Config &cfg, const char *dir);

    inline void addStepLogs(Span<LogEntries> step_data, CountT num_worlds);
    inline void saveBuffersToDisk();

    inline void diskWriterLoop();
};

static inline void writeU32(std::fstream &f, uint32_t v)
{
    f.write((char *)&v, sizeof(uint32_t));
}

StateLogReader::Impl::init(const Config &cfg, const char *dir)
{
    std::filesystem::path log_dir(dir);
    if (!std::filesystem::exists(log_dir)) {
        FATAL("State log directory %s not found", dir);
    }

    HeapArray<StateLogFile> files(cfg.numBytesPerLogType.size());
    for (CountT i = 0; i < files.size(); i++) {
        std::fstream log_file(log_dir / std::filesystem::path(
            std::to_string(i) + ".log"),
            std::ios::binary | std::ios::out);

        new (&files[i]) StateLogFile {
            .hdl = std::move(log_file),
            .numBytesPerEntry = cfg.numBytesPerLogType[i],
        };
    }

    std::fstream index_file(log_dir / std::filesystem::path("index"),
        std::ios::binary | std::ios::out);

    return new Impl {
        .logDir = std::move(log_dir),
        .indexFile = std::move(index_file),
        .logFiles = std::move(files),
    };
}

StateLogWriter::Impl::init(const Config &cfg, const char *dir)
{
    std::filesystem::path log_dir(dir);
    if (std::filesystem::exists(log_dir)) {
        FATAL("State log directory %s already exists", dir);
    }

    bool created_log_dir = std::filesystem::create_directory(log_dir);
    if (!created_log_dir) {
        FATAL("Failed to create directory %s", dir);
    }

    HeapArray<LogWriterFile> files(cfg.numBytesPerLogType.size());
    for (CountT i = 0; i < files.size(); i++) {
        std::fstream log_file(log_dir / std::filesystem::path(
            std::to_string(i) + ".log"),
            std::ios::binary | std::ios::out);

        new (&files[i]) LogWriterFile {
            .hdl = std::move(log_file),
            .buffer = { { nullptr, nullptr }, { 0, 0 }, { 0, 0 } },
            .numBytesPerEntry = cfg.numBytesPerLogType[i],
        };
    }

    std::fstream index_file(log_dir / std::filesystem::path("index"),
        std::ios::binary | std::ios::out);

    return new Impl {
        .logDir = std::move(log_dir),
        .indexFile = std::move(index_file),
        .indexBuffer = { { nullptr, nullptr }, { 0, 0 }, { 0, 0 } },
        .logFiles = std::move(files),
        .curSaveBuffer = 0,
        .numBufferedSteps = { 0, 0 },
        .numStepsBeforeFlush = cfg.numBufferedSteps,
        .diskWriteWakeup = 0,
        .diskWriteThread = std::thread(),
    };
}

void StateLogWriter::Impl::addStepLogs(
    Span<LogEntries> step_data, CountT num_worlds)
{
    uint32_t cur_buffer = curSaveBuffer;



    numBufferSteps += 1;
    if (numBufferedSteps == numStepsBeforeFlush) {
        saveBuffersToDisk();
    }
}

void StateLogWriter::Impl::saveBuffersToDisk()
{
    diskWriteWakeup.wait<sync::acquire>(1);

    curSaveBuffer = curSaveBuffer ^ 1;

    diskWriteWakeup.store_release(1);
    diskWriteWakeup.notify_one();

    for (CountT log_type_idx = 0; log_type_idx < logFiles.size();
         log_type_idx++) {
        logFiles[log_type_idx].buffer.numSavedBytes[curSaveBuffer] = 0;
    }
    numBufferSteps[curSaveBuffer] = 0;
}

void StateLogWriter::Impl::diskWriterLoop()
{
    while (true) {
        diskWriteWakeup.wait<sync::relaxed>(0);
        uint32_t ctrl = diskWriteWakeup.load_acquire();

        if (ctrl == 0) {
            continue;
        } else if (ctrl == -1) {
            break;
        }

        uint32_t cur_buffer = curSaveBuffer ^ 1;

        auto &idx_file = impl_->indexFile;

        writeU32(idx_file, (uint32_t)num_worlds);
        for (CountT log_type_idx = 0; log_type_idx < step_data.size();
             log_type_idx++) {
            const LogEntries &new_entries = step_data[log_type_idx];

            writeU32(idx_file, new_entries.numTotalEntries);

            idx_file.write((char *)new_entries.worldOffsets,
                           sizeof(uint32_t) * (uint64_t)num_worlds);
            idx_file.write((char *)new_entries.worldCounts,
                           sizeof(uint32_t) * (uint64_t)num_worlds);

            auto &log_file = impl_->logFiles[log_type_idx];
            log_file.hdl.write((char *)new_entries.data,
                (uint64_t)log_file.numBytesPerEntry *
                (uint64_t)new_entries.numTotalEntries);
        }

        diskWriteWakeup.store_release(0);
        diskWriteWakeup.notify_one();
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
    impl_->diskWriteThread = std::thread([impl_]() {
        impl_->diskWriterLoop();
    });
}

StateLogWriter::StateLogWriter(StateLogWriter &&) = default;
StateLogWriter::~StateLogWriter()
{
    impl_->diskWriteThread.join();
}

StateLogWriter & StateLogWriter::operator=(StateLogWriter &&) = default;

void StateLogWriter::addStepLogs(Span<LogEntries> step_data, CountT num_worlds)
{
    impl_->addStepLogs(step_data, num_worlds);
}

}
