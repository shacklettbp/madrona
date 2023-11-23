namespace madrona::render {

inline RenderContextFlags & operator|=(RenderContextFlags &a, RenderContextFlags b)
{
    a = RenderContextFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline RenderContextFlags operator|(RenderContextFlags a, RenderContextFlags b)
{
    a |= b;

    return a;
}

inline RenderContextFlags & operator&=(RenderContextFlags &a, RenderContextFlags b)
{
    a = RenderContextFlags(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    return a;
}

inline RenderContextFlags operator&(RenderContextFlags a, RenderContextFlags b)
{
    a &= b;

    return a;
}

}
