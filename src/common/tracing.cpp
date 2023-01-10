#include "madrona/tracing.hpp"

#include <assert.h>
#include <unistd.h>
#include <fstream>
#include <algorithm>

namespace madrona
{
    template <typename T>
    void WriteToFile(T *events, size_t size, const std::string file_path, const std::string name)
    {
        std::string file_name = file_path + std::to_string(getpid()) + name + ".bin";
        std::ofstream myFile(file_name, std::ios::out | std::ios::binary);
        myFile.write((char *)events, size * sizeof(T));
        myFile.close();
    }

    thread_local HostTracing HOST_TRACING;
    void FinalizeLogging(const std::string file_path)
    {
        assert(HOST_TRACING.events.size() == HOST_TRACING.time_stamps.size());
        std::vector<int64_t> concat;
        std::transform(HOST_TRACING.events.begin(), HOST_TRACING.events.end(),
                       std::back_inserter(concat),
                       [](auto e)
                       { return static_cast<int64_t>(e); });

        concat.insert(concat.end(), HOST_TRACING.time_stamps.begin(), HOST_TRACING.time_stamps.end());

        WriteToFile<int64_t>(&concat[0], concat.size(), file_path, "_madrona_host_tracing");
    }
} // namespace madrona
