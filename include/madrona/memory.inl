namespace madrona {

PolyAlloc::PolyAlloc(void *state,
                     void *(*alloc_ptr)(void *, size_t),
                     void (*dealloc_ptr)(void *, void *))
    : state_(state),
      alloc_ptr_(alloc_ptr),
      dealloc_ptr_(dealloc_ptr)
{}

void *PolyAlloc::alloc(size_t num_bytes)
{
    return alloc_ptr_(state_, num_bytes);
}

void PolyAlloc::dealloc(void *ptr)
{
    dealloc_ptr_(state_, ptr);
}

template <typename A>
PolyAlloc Allocator<A>::getPoly()
{
    return PolyAlloc(this, &allocStatic, &deallocStatic);
}

template <typename A>
void * Allocator<A>::allocStatic(void *state, size_t num_bytes)
{
    return static_cast<A *>(state)->alloc(num_bytes);
}

template <typename A>
void Allocator<A>::deallocStatic(void *state, void *ptr)
{
    static_cast<A *>(state)->dealloc(ptr);
}

void * AllocContext::alloc(size_t num_bytes)
{
    return cur_alloc_.alloc(num_bytes);
}

void AllocContext::dealloc(void *ptr)
{
    cur_alloc_.dealloc(ptr);
}

template <typename T, typename ...Args>
T * AllocContext::make(Args &&...args)
{
    auto ptr = (T *)alloc(sizeof(T));
    new (ptr) T(std::forward(args)...);

    return ptr;
}

template <typename T>
void AllocContext::destroy(T *ptr)
{
    ptr->~T();
    dealloc(ptr);
}

template <typename A>
inline AllocScope AllocContext::scope(A &alloc)
{
    return AllocScope(alloc.getPoly(), cur_scope_, this);
}

template <typename A, typename Fn, typename ...Args>
auto AllocContext::with(A &alloc, Fn &&fn, Args &&...args) ->
    decltype(Fn(args...))
{
    AllocScope tracker = scope(alloc);
    return fn(std::forward(args)...);
}

void * DefaultAlloc::alloc(size_t num_bytes)
{
    return malloc(num_bytes);
}

void DefaultAlloc::dealloc(void *ptr)
{
    free(ptr);
}

}
