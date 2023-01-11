#include <madrona/tracing.hpp>

#include <assert.h>

namespace madrona
{
    thread_local HostTracing HOST_TRACING;

    void FinalizeLogging(const std::string file_path)
    {
        auto num_events = HOST_TRACING.events.size();
        assert(num_events == HOST_TRACING.time_stamps.size());
        int64_t concat[num_events * 2];
        for (size_t i = 0; i < num_events; i++)
        {
            concat[i] = static_cast<int64_t>(HOST_TRACING.events[i]);
            concat[i + num_events] = HOST_TRACING.time_stamps[i];
        }

        WriteToFile<int64_t>(concat, num_events * 2, file_path, "_madrona_host_tracing");
    }
} // namespace madrona
