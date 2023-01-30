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
#include <version>

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

namespace sync {
using memory_order = std::memory_order;
inline constexpr memory_order relaxed = std::memory_order_relaxed;
inline constexpr memory_order acquire = std::memory_order_acquire;
inline constexpr memory_order release = std::memory_order_release;
inline constexpr memory_order acq_rel = std::memory_order_acq_rel;
inline constexpr memory_order seq_cst = std::memory_order_seq_cst;
}

using AtomicU32 = std::atomic_uint32_t;
using AtomicI32 = std::atomic_int32_t;
using AtomicU64 = std::atomic_uint64_t;
using AtomicI64 = std::atomic_int64_t;
using AtomicFloat = std::atomic<float>;
static_assert(AtomicFloat::is_always_lock_free);

template <typename T>
class AtomicRef {
public:
    AtomicRef(T &ref)
#ifndef __cpp_lib_atomic_ref
        : addr_(&ref)
#else
        : ref_(ref)
#endif
    {}

    template <sync::memory_order order>
    inline T load() const
    {
#ifndef __cpp_lib_atomic_ref
        return __builtin_bit_cast(T,
            __atomic_load_n((ValueT *)addr_, OrderMap<order>::builtin));
#else
        return ref_.load(order);
#endif
    }

    template <sync::memory_order order>
    inline void store(T v) const
    {
#ifndef __cpp_lib_atomic_ref
        __atomic_store_n((ValueT *)addr_, __builtin_bit_cast(ValueT, v),
                        OrderMap<order>::builtin);
#else
        return ref_.store(v, order);
#endif
    }

    template <sync::memory_order success_order,
              sync::memory_order failure_order>
    inline bool compare_exchange_weak(T &expected, T desired)
    {
#ifndef __cpp_lib_atomic_ref
        return __atomic_compare_exchange_n(
            (ValueT *)addr_, (ValueT *)&expected,
            __builtin_bit_cast(ValueT, desired), true,
            OrderMap<success_order>::builtin,
            OrderMap<failure_order>::builtin);
#else
        return ref_.compare_exchange_weak(expected, desired,
                                          success_order, failure_order);
#endif
    }

    template <sync::memory_order order>
    inline T fetch_add(T v) requires std::is_integral_v<T>
    {
#ifndef __cpp_lib_atomic_ref
        return __atomic_fetch_add(addr_, v, OrderMap<order>::builtin);
#else
        return ref_.fetch_add(v, order);
#endif
    }

    template <sync::memory_order order>
    inline T fetch_sub(T v) requires std::is_integral_v<T>
    {
#ifndef __cpp_lib_atomic_ref
        return __atomic_fetch_sub(addr_, v, OrderMap<order>::builtin);
#else
        return ref_.fetch_sub(v, order);
#endif
    }

private:
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    static_assert(std::is_trivially_copyable_v<T>);

#ifndef __cpp_lib_atomic_ref
    template <size_t t_size> struct ValueType;
    template <> struct ValueType<8> {
        using type = uint64_t;
    };
    template <> struct ValueType<4> {
        using type = uint32_t;
    };

    template <sync::memory_order order> struct OrderMap;
    template <> struct OrderMap<sync::relaxed> {
        static inline constexpr int builtin = __ATOMIC_RELAXED;
    };

    template <> struct OrderMap<sync::acquire> {
        static inline constexpr int builtin = __ATOMIC_ACQUIRE;
    };

    template <> struct OrderMap<sync::release> {
        static inline constexpr int builtin = __ATOMIC_RELEASE;
    };

    template <> struct OrderMap<sync::acq_rel> {
        static inline constexpr int builtin = __ATOMIC_ACQ_REL;
    };

    template <> struct OrderMap<sync::seq_cst> {
        static inline constexpr int builtin = __ATOMIC_SEQ_CST;
    };

    using ValueT = typename ValueType<sizeof(T)>::type;
    T *addr_;
#else
    std::atomic_ref<T> ref_;
    static_assert(decltype(ref_)::is_always_lock_free);
#endif
};

using AtomicI32Ref = AtomicRef<int32_t>;
using AtomicU32Ref = AtomicRef<uint32_t>;
using AtomicI64Ref = AtomicRef<int64_t>;
using AtomicU64Ref = AtomicRef<uint64_t>;
using AtomicFloatRef = AtomicRef<float>;

class 
#ifndef MADRONA_GPU_MODE
    alignas(MADRONA_CACHE_LINE) 
#endif
    SpinLock {
public:
    void lock()
    {
        while (lock_.exchange(1, sync::acquire) == 1) {
            while (lock_.load(sync::relaxed) == 1) {}
        }
    }

    // Test and test-and-set
    bool tryLock()
    {
        int32_t is_locked = lock_.load(sync::relaxed);
        if (is_locked == 1) return false;

        int32_t prev_locked = lock_.exchange(1, sync::relaxed);

        if (prev_locked) {
            return false;
        }

        std::atomic_thread_fence(sync::acquire);
        TSAN_ACQUIRE(&lock_);

        return true;
    }

    void unlock()
    {
        lock_.store(0, sync::release);
    }

private:
    AtomicI32 lock_ { false };
};

}
