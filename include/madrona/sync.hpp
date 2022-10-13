/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/macros.hpp>

#include <atomic>

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define TSAN_ENABLED (1)
#endif
#endif

#ifdef TSAN_ENABLED
extern "C" {
extern void __tsan_acquire(void *addr);
extern void __tsan_release(void *addr);
}

#define TSAN_ACQUIRE(addr) __tsan_acquire(addr)
#define TSAN_RELEASE(addr) __tsan_release(addr)
#define TSAN_DISABLED __attribute__((no_sanitize_thread))

#else
#define TSAN_ACQUIRE(addr)
#define TSAN_RELEASE(addr)
#define TSAN_DISABLED
#endif

namespace madrona {
namespace utils {

class alignas(MADRONA_CACHE_LINE) SpinLock {
public:
    void lock()
    {
        while (lock_.test_and_set(std::memory_order_acquire)) {
            while (lock_.test(std::memory_order_relaxed)) {}
        }
    }

    // Test and test-and-set
    bool tryLock()
    {
        bool is_locked = lock_.test(std::memory_order_relaxed);
        if (is_locked) return false;

        bool prev_locked = lock_.test_and_set(std::memory_order_relaxed);

        if (prev_locked) {
            return false;
        }

        std::atomic_thread_fence(std::memory_order_acquire);
        TSAN_ACQUIRE(&lock_);

        return true;
    }

    void unlock()
    {
        lock_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag lock_ { false };
};

}
}
