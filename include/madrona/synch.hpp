#pragma once

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
extern void AnnotateHappensBefore(const char *file, int line,
                                  const volatile void *cv);
extern void AnnotateHappensAfter(const char *file, int line,
                                 const volatile void *cv);
}

#define TSAN_ACQUIRE(addr) __tsan_acquire(addr)
#define TSAN_RELEASE(addr) __tsan_release(addr)
#define TSAN_HAPPENS_BEFORE(addr) \
    AnnotateHappensBefore(__FILE__, __LINE__, addr)
#define TSAN_HAPPENS_AFTER(addr) \
    AnnotateHappensAfter(__FILE__, __LINE__, addr)

#else
#define TSAN_ACQUIRE(addr)
#define TSAN_RELEASE(addr)
#define TSAN_HAPPENS_BEFORE(addr)
#define TSAN_HAPPENS_AFTER(addr)
#endif

namespace madrona {
namespace utils {

class SpinLock {
public:
    void lock()
    {
        while (lock_.test_and_set(std::memory_order_acquire)) {
            while (lock_.test(std::memory_order_relaxed)) {}
        }
    }

    bool tryLock()
    {
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
