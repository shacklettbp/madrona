#include <madrona/state_log.hpp>
#include <madrona/heap_array.hpp>

#include <filesystem>
#include <fstream>

namespace madrona {

struct StateLogFile {
    std::fstream hdl;
    uint32_t numBytesPerEntry;
};

struct StateLogStore::Impl {
    std::filesystem::path logDir;
    std::fstream indexFile;
    HeapArray<StateLogFile> logFiles;
};

static inline void writeU32(std::fstream &f, uint32_t v)
{
    f.write((char *)&v, sizeof(uint32_t));
}

StateLogStore::StateLogStore(StateLogStore &&) = default;
StateLogStore::~StateLogStore() = default;
StateLogStore & StateLogStore::operator=(StateLogStore &&) = default;

StateLogStore::StateLogStore(Impl *impl)
    : impl_(impl)
{}

StateLogStore StateLogStore::initNewStateLog(
    const Config &cfg,
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

    writeU32(index_file, cfg.numBytesPerLogType.size());

    for (CountT i = 0; i < files.size(); i++) {
        writeU32(index_file, files[i].numBytesPerEntry);
    } 

    Impl *impl = new Impl {
        .logDir = std::move(log_dir),
        .indexFile = std::move(index_file),
        .logFiles = std::move(files),
    };

    return StateLogStore(impl);
}

void StateLogStore::addStepLogs(Span<LogEntries> step_data, CountT num_worlds)
{
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
}

}
