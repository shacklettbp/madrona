namespace madrona::cu {

void checkNVRTC(nvrtcResult res, const char *file,
                int line, const char *funcname) noexcept
{
    if (res != NVRTC_SUCCESS) {
        nvrtcError(res, file, line, funcname);
    }
}

void checknvJitLink(nvJitLinkResult res, const char *file,
                    int line, const char *funcname) noexcept
{
    if (res != NVJITLINK_SUCCESS) {
        nvJitLinkError(res, file, line, funcname);
    }
}


}
