namespace madrona::imp {

inline SourceTexture::SourceTexture()
{
}

inline SourceTexture::SourceTexture(const char *path_ptr) :
    info(TextureLoadInfo::FileName), path(path_ptr)
{
}

inline SourceTexture::SourceTexture(
        TextureLoadInfo tex_info, const char *path_ptr)
{
    info = tex_info;
    path = path_ptr;
}

inline SourceTexture::SourceTexture(PixelBufferInfo p_info) 
{
    info = TextureLoadInfo::PixelBuffer;
    pix_info = p_info;
}

}
