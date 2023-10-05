namespace madrona {

template <typename... ComponentTs>
inline ComponentMetadataSelector<ComponentTs...>::ComponentMetadataSelector(
        ComponentFlags component_flags)
{
    flags.fill(component_flags);
}

template <typename... ComponentTs>
template <typename... FlagTs>
inline ComponentMetadataSelector<ComponentTs...>::ComponentMetadataSelector(
        FlagTs ...in_flags)
    : flags { in_flags ... }
{}

inline ArchetypeFlags & operator|=(ArchetypeFlags &a, ArchetypeFlags b)
{
    a = ArchetypeFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline ArchetypeFlags operator|(ArchetypeFlags a, ArchetypeFlags b)
{
    a |= b;

    return a;
}

inline ArchetypeFlags & operator&=(ArchetypeFlags &a, ArchetypeFlags b)
{
    a = ArchetypeFlags(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    return a;
}

inline ArchetypeFlags operator&(ArchetypeFlags a, ArchetypeFlags b)
{
    a &= b;

    return a;
}

inline ComponentFlags & operator|=(ComponentFlags &a, ComponentFlags b)
{
    a = ComponentFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline ComponentFlags operator|(ComponentFlags a, ComponentFlags b)
{
    a |= b;

    return a;
}

inline ComponentFlags & operator&=(ComponentFlags &a, ComponentFlags b)
{
    a = ComponentFlags(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    return a;
}

inline ComponentFlags operator&(ComponentFlags a, ComponentFlags b)
{
    a &= b;

    return a;
}

}
