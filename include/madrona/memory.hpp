/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace madrona {

// Virtual Adapter
class PolyAlloc {
public:
    inline PolyAlloc(void *state,
                     void *(*alloc_ptr)(void *, size_t),
                     void (*dealloc_ptr)(void *, void *));
    inline void * alloc(size_t num_bytes);
    inline void dealloc(void *ptr);

private:
    void *state_;
    void *(*alloc_ptr_)(void *, size_t);
    void (*dealloc_ptr_)(void *, void *);
};

class AllocContext;

class AllocScope {
public:
    AllocScope(const AllocScope &) = delete;
    ~AllocScope();

private:
    AllocScope(const PolyAlloc &alloc, AllocScope *parent,
               AllocContext *ctx);

    PolyAlloc cur_alloc_;
    AllocScope *parent_;
    AllocContext *ctx_;

friend class AllocContext;
};

class AllocContext {
public:
    AllocContext();
    AllocContext(const AllocContext &) = delete;
    
    // Allocation functions
    inline void * alloc(size_t num_bytes);
    inline void dealloc(void *ptr);

    template <typename T, typename ...Args>
    inline T * make(Args &&...args);

    template <typename T>
    inline void destroy(T *);

    // Scoped helpers
    template <typename A>
    inline AllocScope scope(A &alloc);

    template <typename A, typename Fn, typename ...Args>
    auto with(A &alloc, Fn &&fn, Args &&...args) ->
        decltype(Fn(args...));

    inline const PolyAlloc &getAlloc() { return cur_alloc_; }

private:
    PolyAlloc cur_alloc_;
    AllocScope *cur_scope_;

friend class AllocScope;
};

// Allocator base class, CRTP for polymorphic adapter
template <typename A>
class Allocator {
public:
    inline PolyAlloc getPoly();

private:
    static inline void * allocStatic(void *state, size_t num_bytes);
    static inline void deallocStatic(void *state, void *ptr);
};

class DefaultAlloc : public Allocator<DefaultAlloc> {
public:
    inline void * alloc(size_t num_bytes);
    inline void dealloc(void *ptr);
};

// FIXME:
using InitAlloc = DefaultAlloc;
using TmpAlloc = DefaultAlloc;

}

#include "memory.inl"
