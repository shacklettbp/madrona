#pragma once

#include <atomic>
#include <array>

#include <madrona/macros.hpp>

struct WorkItem {
    uint32_t job_id;
    void *data;
};

template <int queue_size>
class WorkQueue {
public:
    inline void enqueue(WorkItem work_item)
    {
        do {
            uint32_t cur_count = count_.load_relaxed();
            if (cur_count >= queue_size) {
                continue;
            }

            uint32_t cur_head = head_.load_relaxed();
            uint32_t cur_tail = tail_.load_relaxed();
        } while(true);
        while (!ensureEnqueue()) {

            
        }
    }

private:
    MADRONA_ALWAYS_INLINE bool ensureEnqueue()
    {
    }

    std::array<AtomicU32, queue_size> tickets_;
    AtomicU32 head_;
    AtomicU32 tail_;
    AtomicI32 count_;
};
