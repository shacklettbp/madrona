namespace madrona::imp {

inline SourceTexture::SourceTexture() :
        imageData(0)
{
}

inline SourceTexture::SourceTexture(const char *path_ptr) :
    imageData(0)
{
    config = {};
    config.format = TextureFormat::PNG;
}

inline SourceTexture::SourceTexture(
        TextureLoadInfo tex_info, const char *path_ptr) :
        imageData(0)
{
    config = {};
    config.format = TextureFormat::PNG;
}

inline SourceTexture::SourceTexture(PixelBufferInfo p_info) :
imageData(0)
{
    config = {};
    config.format = TextureFormat::PNG;
}

}
