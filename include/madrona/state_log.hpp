#pragma once

#include <madrona/span.hpp>

#include <cstdint>
#include <memory>

namespace madrona {

class StateLogStore {
public:
    struct Config {
        Span<const uint32_t> numBytesPerLogType;
    };

    struct LogEntries {
        void *data;
        uint32_t *worldOffsets;
        uint32_t *worldCounts;
        uint32_t numTotalEntries;
    };

    StateLogStore(StateLogStore &&);
    ~StateLogStore();
    StateLogStore & operator=(StateLogStore &&);

    static StateLogStore initNewStateLog(
        const Config &cfg, const char *dir);

    void addStepLogs(Span<LogEntries> step_data, CountT num_worlds);

private:
    struct Impl;
    StateLogStore(Impl *impl);
    std::unique_ptr<Impl> impl_;
};

}
