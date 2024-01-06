#pragma once

#include <madrona/span.hpp>

#include <cstdint>
#include <memory>

namespace madrona {

class StateLogReader {
public:
    struct Config {
        Span<const uint32_t> numBytesPerLogType;
    };

    StateLogReader(const Config &cfg, const char *dir);
    StateLogReader(StateLogReader &);
    ~StateLogReader();
    StateLogReader & operator=(StateLogReader &&);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};


class StateLogWriter {
public:
    struct Config {
        Span<const uint32_t> numBytesPerLogType;
        uint32_t numBufferSteps;
        uint32_t maxNumStepsSaved;
    };

    struct LogEntries {
        void *data;
        uint32_t *worldOffsets;
        uint32_t *worldCounts;
        uint32_t numTotalEntries;
    };

    StateLogWriter(const Config &cfg, const char *dir);

    StateLogWriter(StateLogWriter &&);
    ~StateLogWriter();
    StateLogWriter & operator=(StateLogWriter &&);

    void addStepLogs(Span<LogEntries> step_data, CountT num_worlds);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
