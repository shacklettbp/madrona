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
            uint32_t cur_count = count_.load(std::memory_order_relaxed);
            if (cur_count >= queue_size) {
                continue;
            }

            uint32_t cur_head = head_.load(std::memory_order_relaxed);
            uint32_t cur_tail = tail_.load(std::memory_order_relaxed);
        } while(true);
        while (!ensureEnqueue()) {

            
        }
    }

private:
    MADRONA_ALWAYS_INLINE bool ensureEnqueue()
    {
    }

    std::array<std::atomic_uint32_t, queue_size> tickets_;
    std::atomic_uint32_t head_;
    std::atomic_uint32_t tail_;
    std::atomic_int32_t count_;
};
