#include <madrona/importer.hpp>
#include <madrona/io.hpp>

#include <filesystem>
#include <string>
#include <unordered_map>

#include <stb_image.h>

namespace madrona::imp {

namespace {

enum class DefaultTypeCode : int32_t {
    PNG,
    JPG,
    NumDefault,
};

}

struct ImageImporter::Impl {
    std::unordered_map<std::string, int32_t> extensionToTypeCode;
    std::unordered_map<int32_t, ImportHandler> typeCodeToHandler;
    int32_t nextTypeCode;

    static inline Impl * make();
};

static Optional<SourceTexture> stbiImportR8G8B8A8(void *data, size_t num_bytes)
{
    int width, height, channels;
    uint8_t *pixel_data = stbi_load_from_memory((uint8_t *)data,
        (int)num_bytes, &width, &height, &channels, STBI_rgb_alpha);

    if (!pixel_data) {
        return Optional<SourceTexture>::none();
    }

    return SourceTexture {
        .data = pixel_data,
        .format = SourceTextureFormat::R8G8B8A8,
        .width = (uint32_t)width,
        .height = (uint32_t)height,
        .numBytes = (size_t)width * (size_t)height * 4,
    };
}

ImageImporter::Impl * ImageImporter::Impl::make()
{
    return new Impl {
        .extensionToTypeCode = {
            { "png", (int32_t)DefaultTypeCode::PNG },
            { "jpg", (int32_t)DefaultTypeCode::JPG },
        },
        .typeCodeToHandler = {
            { (int32_t)DefaultTypeCode::PNG, &stbiImportR8G8B8A8 },
            { (int32_t)DefaultTypeCode::JPG, &stbiImportR8G8B8A8 },
        },
        .nextTypeCode = (int32_t)DefaultTypeCode::NumDefault,
    };
}

ImageImporter::ImageImporter()
    : impl_(Impl::make())
{}

ImageImporter::ImageImporter(ImageImporter &&) = default;
ImageImporter::~ImageImporter() = default;

int32_t ImageImporter::addHandler(const char *extension, ImportHandler fn)
{
    int32_t type_code = impl_->nextTypeCode++;

    impl_->extensionToTypeCode.emplace(extension, type_code);
    impl_->typeCodeToHandler.emplace(type_code, fn);

    return type_code;
}

int32_t ImageImporter::getPNGTypeCode()
{
    return (int32_t)DefaultTypeCode::PNG;
}

int32_t ImageImporter::getJPGTypeCode()
{
    return (int32_t)DefaultTypeCode::JPG;
}

int32_t ImageImporter::getExtensionTypeCode(const char *extension)
{
    auto type_code_iter = impl_->extensionToTypeCode.find(extension);

    if (type_code_iter == impl_->extensionToTypeCode.end()) {
        return -1;
    }

    return type_code_iter->second;
}

Optional<SourceTexture> ImageImporter::importImage(
    void *data, size_t num_bytes, int32_t type_code)
{
    ImportHandler handler = impl_->typeCodeToHandler.find(type_code)->second;

    return handler(data, num_bytes);
}

Optional<SourceTexture> ImageImporter::importImage(const char *path)
{
    std::string extension = std::filesystem::path(path).extension().string();

    // Extension contains the leading .
    int32_t type_code = getExtensionTypeCode(extension.c_str() + 1);

    if (type_code == -1) {
        return Optional<SourceTexture>::none();
    }

    size_t num_bytes;
    char *file_data = readBinaryFile(path, 1, &num_bytes);

    Optional<SourceTexture> img = importImage(file_data, num_bytes, type_code);

    rawDeallocAligned(file_data);

    return img;
}

Span<SourceTexture> ImageImporter::importImages(
    StackAlloc &tmp_alloc,
    Span<const char * const> paths)
{
    SourceTexture *out_textures = tmp_alloc.allocN<SourceTexture>(paths.size());

    for (CountT i = 0; i < paths.size(); i++) {
        const char *path = paths[i];
        Optional<SourceTexture> tex = importImage(path);
        if (!tex.has_value()) {
            return Span<SourceTexture>(nullptr, 0);
        }

        out_textures[i] = *tex;
    }

    return Span(out_textures, paths.size());
}

void ImageImporter::deallocImportedImages(Span<SourceTexture> textures)
{
    for (SourceTexture &tex : textures) {
        free(tex.data);
    }
}

}
