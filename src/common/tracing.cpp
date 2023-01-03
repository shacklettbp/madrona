#include "madrona/tracing.hpp"

#include <assert.h>
#include <unistd.h>
#include <fstream>
#include <algorithm>

namespace madrona
{
    thread_local HostTracing HOST_TRACING;
    void FinalizeLogging(const std::string file_path)
    {
        assert(HOST_TRACING.events.size() == HOST_TRACING.time_stamps.size());
        std::vector<int64_t> concat;
        std::transform(HOST_TRACING.events.begin(), HOST_TRACING.events.end(),
                       std::back_inserter(concat),
                       [](auto e)
                       { return static_cast<int>(e); });

        std::string file_name = file_path + std::to_string(getpid()) + "_madrona_host_tracing.bin";
        std::ofstream myFile(file_name, std::ios::out | std::ios::binary);
        myFile.write((char *)&concat[0], concat.size() * sizeof(int64_t));
        myFile.close();
    }
} // namespace madrona
