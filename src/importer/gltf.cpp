#include "gltf.hpp"

#include <madrona/crash.hpp>
#include <madrona/json.hpp>
#include <madrona/math.hpp>
#include <madrona/optional.hpp>

#include <cstdint>
#include <cstdarg>
#include <optional>
#include <filesystem>
#include <string_view>
#include <vector>
#include <fstream>
#include <type_traits>

using std::string;
using std::string_view;
using std::is_same_v;
using std::conditional_t;
using std::is_const_v;
using namespace simdjson;

namespace madrona::imp {

namespace {

template <typename T>
class GLTFStridedSpan {
public:
    using RawPtrType = conditional_t<is_const_v<T>,
                                     const uint8_t *,
                                     uint8_t *>; 

    GLTFStridedSpan(RawPtrType data, size_t num_elems, size_t byte_stride)
        : raw_data_(data),
          num_elems_(num_elems),
          byte_stride_(byte_stride)
    {}

    constexpr const T& operator[](size_t idx) const
    {
        return *fromRaw(raw_data_ + idx * byte_stride_);
    }

    constexpr T& operator[](size_t idx)
    {
        return *fromRaw(raw_data_ + idx * byte_stride_);
    }

    T *data() { return fromRaw(raw_data_); }
    const T *data() const { return fromRaw(raw_data_); }

    constexpr size_t size() const { return num_elems_; }

    template <typename U>
    class IterBase {
    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = U;
        using difference_type = std::ptrdiff_t;
        using pointer = U *;
        using reference = U &;

        IterBase(RawPtrType ptr, size_t byte_stride)
            : ptr_(ptr),
              byte_stride_(byte_stride)
        {}

        IterBase& operator+=(difference_type d)
        {
            ptr_ += d * byte_stride_;
            return *this;
        }

        IterBase& operator-=(difference_type d)
        {
            ptr_ -= d * byte_stride_;
            return *this;
        }

        friend IterBase operator+(IterBase it, difference_type d)
        {
            it += d;
            return it;
        }

        friend IterBase operator+(difference_type d, IterBase it)
        {
            return it + d;
        }

        friend IterBase operator-(IterBase it, difference_type d)
        {
            it -= d;
            return it;
        }

        friend difference_type operator-(const IterBase &a,
                                         const IterBase &b)
        {
            assert(a.byte_stride_ == b.byte_stride_);
            return (a.ptr_ - b.ptr_) / a.byte_stride_;
        }

        bool operator==(IterBase o) const { return ptr_ == o.ptr_; }
        bool operator!=(IterBase o) const { return !(*this == o); }

        reference operator[](difference_type d) const
        {
            return *(*this + d);
        }
        reference operator*() const
        {
            return *fromRaw(ptr_);
        }

        friend bool operator<(const IterBase &a, const IterBase &b)
        {
            return a.ptr_ < b.ptr_;
        }

        friend bool operator>(const IterBase &a, const IterBase &b)
        {
            return a.ptr_ > b.ptr_;
        }

        friend bool operator<=(const IterBase &a, const IterBase &b)
        {
            return !(a > b);
        }

        friend bool operator>=(const IterBase &a, const IterBase &b)
        {
            return !(a < b);
        }

        IterBase &operator++() { *this += 1; return *this; };
        IterBase &operator--() { *this -= 1; return *this; };

                IterBase operator++(int)
        {
            IterBase t = *this;
            operator++();
            return t;
        }
        IterBase operator--(int)
        {
            IterBase t = *this;
            operator--();
            return t;
        }
    private:
        RawPtrType ptr_;
        size_t byte_stride_;
    };

    using iterator = IterBase<T>;
    using const_iterator = IterBase<const T>;

    iterator begin()
    {
        return iterator(raw_data_, byte_stride_);
    }
    iterator end()
    {
        return iterator(raw_data_ + num_elems_ * byte_stride_, byte_stride_);
    }

    const_iterator begin() const
    {
        return const_iterator(raw_data_, byte_stride_);
    }
    const_iterator end() const
    {
        return const_iterator(raw_data_ + num_elems_ * byte_stride_,
                              byte_stride_);
    }

    bool contiguous() const { return byte_stride_ == value_size_; }

private:
    RawPtrType raw_data_;
    size_t num_elems_;
    size_t byte_stride_;

    static constexpr size_t value_size_ = sizeof(T);

    static RawPtrType toRaw(T *ptr)
    {
        if constexpr (std::is_same_v<T *, RawPtrType>) {
            return ptr;
        } else {
            return reinterpret_cast<RawPtrType>(ptr);
        }
    }

    static T * fromRaw(RawPtrType ptr)
    {
        if constexpr (std::is_same_v<T *, RawPtrType>) {
            return ptr;
        } else {
            return reinterpret_cast<T *>(ptr);
        }
    }

    friend class IterBase<T>;
};


struct GLBHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t length;
};

struct ChunkHeader {
    uint32_t chunkLength;
    uint32_t chunkType;
};

struct GLTFBuffer {
    const uint8_t *dataPtr;
    std::string_view filePath;
};

struct GLTFBufferView {
    uint32_t bufferIdx;
    uint32_t offset;
    uint32_t stride;
    uint32_t numBytes;
};

enum class GLTFComponentType {
    UINT32,
    UINT16,
    INT16,
    UINT8,
    INT8,
    FLOAT
};

struct GLTFAccessor {
    uint32_t viewIdx;
    uint32_t offset;
    uint32_t numElems;
    GLTFComponentType type;
};
    
enum class GLTFImageType {
    UNKNOWN,
    JPEG,
    PNG,
    BASIS,
    EXTERNAL
};

struct GLTFImage {
    GLTFImageType type;
    union {
        std::string_view filePath;
        uint32_t viewIdx;
    };
};

struct GLTFTexture {
    uint32_t sourceIdx;
    uint32_t samplerIdx;
};

struct GLTFMaterial {
    std::string name;
    uint32_t baseColorIdx;
    uint32_t metallicRoughnessIdx;
    uint32_t specularIdx;
    uint32_t normalIdx;
    uint32_t emittanceIdx;
    uint32_t transmissionIdx;
    uint32_t clearcoatIdx;
    uint32_t clearcoatNormalIdx;
    uint32_t anisoIdx;
    math::Vector4 baseColor;
    float transmissionFactor;
    math::Vector3 baseSpecular;
    float specularFactor;
    float metallic;
    float roughness;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    math::Vector3 attenuationColor;
    float attenuationDistance;
    float anisoScale;
    math::Vector3 anisoDir;
    math::Vector3 baseEmittance;
    bool thinwalled;
};

struct GLTFPrimitive {
    uint32_t positionIdx;
    Optional<uint32_t> normalIdx;
    Optional<uint32_t> uvIdx;
    Optional<uint32_t> colorIdx;
    uint32_t indicesIdx;
    uint32_t materialIdx;
};

struct GLTFMesh {
    std::string name;
    uint32_t primOffset;
    uint32_t numPrims;
};

struct GLTFNode {
    uint32_t childOffset;
    uint32_t numChildren;
    uint32_t meshIdx;
    math::Mat3x4 transform;
};

}

struct GLTFLoader::Impl {
    Span<char> errBuf;
    DynArray<char> jsonBuf;
    ondemand::parser jsonParser;

    const char *curFileName;
    std::filesystem::path sceneDirectory;

    // Scene data filled when loading a GLTF file
    std::string sceneName;
    DynArray<uint8_t> internalData;
    DynArray<GLTFBuffer> buffers;
    DynArray<GLTFBufferView> bufferViews;
    DynArray<GLTFAccessor> accessors;
    DynArray<GLTFImage> images;
    DynArray<GLTFTexture> textures;
    DynArray<GLTFMaterial> materials;
    DynArray<GLTFPrimitive> prims;
    DynArray<GLTFMesh> meshes;
    DynArray<uint32_t> childNodes;
    DynArray<GLTFNode> nodes;
    DynArray<int32_t> rootNodes;

    Impl(Span<char> err_buf);
    void recordError(const char *fmt, ...) const;
    void recordJSONError(error_code err) const;
};

using LoaderData = GLTFLoader::Impl;

template <typename T, CountT num_components>
static bool jsonReadVecImpl(const LoaderData &loader,
                            ondemand::array arr,
                            T &out)
{
    CountT component_idx = 0;
    for (auto comp : arr) {
        auto v = comp.get_double();
        if (v.error()) {
            loader.recordJSONError(v.error());
            return true;
        }

        if (component_idx < num_components) {
            out[component_idx] = float(v.value_unsafe());
        }

        component_idx++;
    }

    if (component_idx != num_components) {
        loader.recordError(
            "Incorrect number of components when parsing Vector");
        return true;
    }

    return false;
}

template <typename T>
static bool jsonReadVec(const LoaderData &loader,
                        ondemand::array arr,
                        T &out)
{
    if constexpr (is_same_v<T, math::Vector4>) {
        return jsonReadVecImpl<T, 4>(loader, arr, out);
    } else if constexpr (is_same_v<T, math::Vector3>) {
        return jsonReadVecImpl<T, 3>(loader, arr, out);
    } else if constexpr (is_same_v<T, math::Vector2>) {
        return jsonReadVecImpl<T, 2>(loader, arr, out);
    }
}

template <typename T, typename U>
static bool jsonGetOr(const LoaderData &loader,
                      simdjson::simdjson_result<U> e,
                      T default_val,
                      T &out)
{
    using ReadType = conditional_t<
        is_same_v<T, float>, double, conditional_t<
        is_same_v<T, uint32_t>, uint64_t, conditional_t<
        is_same_v<T, math::Vector2>, ondemand::array, conditional_t<
        is_same_v<T, math::Vector3>, ondemand::array, conditional_t<
        is_same_v<T, math::Vector4>, ondemand::array, void
    >>>>>;

    static_assert(!is_same_v<ReadType, void>);

    ReadType tmp;
    auto err = e.get(tmp);

    if (!err) {
        if constexpr (is_same_v<ReadType, ondemand::array>) {
            return jsonReadVec<T>(loader, tmp, out);
        } else {
            out = T(tmp);
            return false;
        }
    } else if (err == simdjson::NO_SUCH_FIELD) {
        out = default_val;
        return false;
    } else {
        loader.recordJSONError(err);
        return true;
    }
}

static bool gltfLoad(const char *gltf_filename,
                     LoaderData &loader)
{
    loader.curFileName = gltf_filename;
    std::filesystem::path gltf_path(gltf_filename);

    loader.sceneName = gltf_path.stem().string();
    loader.sceneDirectory = gltf_path.parent_path();

    auto suffix = gltf_path.extension();
    bool binary = suffix == ".glb";

    ondemand::document json_doc;
    if (binary) {
        std::ifstream binary_file(gltf_path.string(),
                                  std::ios::in | std::ios::binary);

        if (!binary_file.is_open() || !binary_file.good()) {
            loader.recordError("Could not open.");
            return false;
        }

        GLBHeader glb_header;
        binary_file.read(reinterpret_cast<char *>(&glb_header),
                         sizeof(GLBHeader));

        uint32_t total_length = glb_header.length;

        ChunkHeader json_header;
        binary_file.read(reinterpret_cast<char *>(&json_header),
                         sizeof(ChunkHeader));

        loader.jsonBuf.resize(json_header.chunkLength + SIMDJSON_PADDING,
                              [](auto *) {});

        binary_file.read(reinterpret_cast<char *>(loader.jsonBuf.data()),
                         json_header.chunkLength);

        auto err = loader.jsonParser.iterate(loader.jsonBuf.data(),
            json_header.chunkLength, loader.jsonBuf.size()).get(json_doc);

        if (err) {
            loader.recordJSONError(err);
            return false;
        }

        if (json_header.chunkLength < total_length) {
            ChunkHeader bin_header;
            binary_file.read(reinterpret_cast<char *>(&bin_header),
                             sizeof(ChunkHeader));

            if (bin_header.chunkType != 0x004E4942) {
                loader.recordError("Invalid bin chunk.");
                return false;
            }

            loader.internalData.resize(bin_header.chunkLength,
                                       [](uint8_t *) {});

            binary_file.read(
                reinterpret_cast<char *>(loader.internalData.data()),
                bin_header.chunkLength);
        }
    } else {
        auto json_data = padded_string::load(gltf_filename);

        if (json_data.error()) {
            loader.recordError("Could not open");
            return false;
        }

        auto err = loader.jsonParser.iterate(json_data).get(json_doc);
        if (err) {
            loader.recordJSONError(err);
            return false;
        }
    }

    auto buffers = json_doc["buffers"].get_array();

    if (buffers.error()) {
        loader.recordJSONError(buffers.error());
        return false;
    }

    for (auto buffer : buffers.value_unsafe()) {
        string_view uri {};
        const uint8_t *data_ptr = nullptr;

        auto uri_elem = buffer["uri"].get_string();
        if (!uri_elem.error()) {
            uri = uri_elem.value_unsafe();
        } else {
            data_ptr = loader.internalData.data();
        }
        loader.buffers.push_back(GLTFBuffer {
            data_ptr,
            uri,
        });
    }

    //cout << "Buffers" << endl;
    
    auto buffer_views = json_doc["bufferViews"].get_array();

    if (buffer_views.error()) {
        loader.recordJSONError(buffer_views.error());
        return false;
    }

    for (auto view : buffer_views.value_unsafe()) {
        auto buffer = view["buffer"].get_uint64();
        if (buffer.error()) {
            loader.recordJSONError(buffer.error());
            return false;
        }

        auto byte_offset = view["byteOffset"].get_uint64();
        if (byte_offset.error()) {
            loader.recordJSONError(byte_offset.error());
            return false;
        }

        uint64_t stride;
        auto stride_err = view["byteStride"].get(stride);
        if (stride_err) {
            stride = 0;
        }

        auto byte_len = view["byteLength"].get_uint64();
        if (byte_len.error()) {
            loader.recordJSONError(byte_len.error());
            return false;
        }

        loader.bufferViews.push_back(GLTFBufferView {
            static_cast<uint32_t>(buffer.value_unsafe()),
            static_cast<uint32_t>(byte_offset.value_unsafe()),
            static_cast<uint32_t>(stride),
            static_cast<uint32_t>(byte_len.value_unsafe()),
        });
    }

    //cout << "bufferViews" << endl;

    auto accessors = json_doc["accessors"].get_array();
    if (accessors.error()) {
        loader.recordJSONError(accessors.error());
        return false;
    }

    for (auto accessor : accessors.value_unsafe()) {
        GLTFComponentType type;
        uint64_t component_type;
        auto component_type_err =
            accessor["componentType"].get(component_type);
        if (component_type_err) {
            loader.recordJSONError(component_type_err);
            return false;
        }

        switch (component_type) {
        case 5126: {
            type = GLTFComponentType::FLOAT;
        } break;
        case 5125: {
            type = GLTFComponentType::UINT32;
        } break;
        case 5123: {
            type = GLTFComponentType::UINT16;
        } break;
        case 5122: {
            type = GLTFComponentType::INT16;
        } break;
        case 5121: {
            type = GLTFComponentType::UINT8;
        } break;
        case 5120: {
            type = GLTFComponentType::INT8;
        } break;
        default: {
            loader.recordError("Unknown component type %" PRIu64,
                               component_type);
            return false;
        } break;
        }

        auto buffer_view_idx = accessor["bufferView"].get_uint64();
        if (buffer_view_idx.error()) {
            loader.recordJSONError(buffer_view_idx.error());
            return false;
        }

        uint64_t byte_offset;
        auto offset_error = accessor["byteOffset"].get(byte_offset);
        if (offset_error) {
            byte_offset = 0;
        }

        auto accessor_count = accessor["count"].get_uint64();
        if (accessor_count.error()) {
            loader.recordJSONError(accessor_count.error());
            return false;
        }

        loader.accessors.push_back(GLTFAccessor {
            static_cast<uint32_t>(buffer_view_idx.value_unsafe()),
            static_cast<uint32_t>(byte_offset),
            static_cast<uint32_t>(accessor_count.value_unsafe()),
            type,
        });
    }

    //cout << "accessors" << endl;
    
    auto images = json_doc["images"].get_array();

    if (!images.error()) {
        for (auto json_image : images.value_unsafe()) {
            GLTFImage img {};
            string_view uri {};
            auto uri_err = json_image["uri"].get(uri);
            if (!uri_err) {
                img.type = GLTFImageType::EXTERNAL;
                img.filePath = uri;
            } else {
                string_view mime;
                auto mime_err = json_image["mimeType"].get(mime);
                if (mime_err) {
                    loader.recordJSONError(mime_err);
                    return false;
                }

                if (mime == "image/jpeg") {
                    img.type = GLTFImageType::JPEG;
                } else if (mime == "image/png") {
                    img.type = GLTFImageType::PNG;
                } else if (mime == "image/x-basis") {
                    img.type = GLTFImageType::BASIS;
                } else {
                    img.type = GLTFImageType::UNKNOWN;
                }

                auto view_idx = json_image["bufferView"].get_uint64();
                img.viewIdx = view_idx.value_unsafe();
            }

            loader.images.push_back(img);
        }
    }

    //cout << "images" << endl;

    auto textures = json_doc["textures"].get_array();
    if (!textures.error()) {
        for (auto texture : textures.value_unsafe()) {
            uint64_t source_idx;
            auto src_err = texture["source"].get(source_idx);
            if (src_err) {
                auto google_ext_err =
                    texture["extensions"]["GOOGLE_texture_basis"]["source"]
                        .get(source_idx);
                if (google_ext_err) {
                    auto khr_ext_err =
                        texture["extensions"]["KHR_texture_basisu"]["source"]
                            .get(source_idx);

                    if (khr_ext_err) {
                        loader.recordError("Texture without source");
                        return false;
                    }
                }
            }

            uint64_t sampler_idx;
            auto sampler_error = texture["sampler"].get(sampler_idx);
            if (sampler_error) {
                sampler_idx = 0;
            }

            loader.textures.push_back(GLTFTexture {
                static_cast<uint32_t>(source_idx),
                static_cast<uint32_t>(sampler_idx),
            });
        }
    }

    //cout << "textures" << endl;
    
    auto materials = json_doc["materials"].get_array();

    if (!materials.error()) {
        for (auto material : materials.value_unsafe()) {
            const uint32_t tex_missing = loader.textures.size();
            auto pbr = material["pbrMetallicRoughness"];
            uint32_t base_color_idx;
            bool mat_err = jsonGetOr(
                loader, pbr["baseColorTexture"]["index"],
                tex_missing, base_color_idx);
            if (mat_err) return false;

            uint32_t metallic_roughness_idx;
            mat_err = jsonGetOr(
                loader, pbr["metallicRoughnessTexture"]["index"],
                tex_missing, metallic_roughness_idx);
            if (mat_err) return false;

            uint32_t bc_coord;
            mat_err = jsonGetOr(
                loader, pbr["baseColorTexture"]["texCoord"],
                0u, bc_coord);
            if (mat_err) return false;

            uint32_t mr_coord;
            mat_err = jsonGetOr(
                loader, pbr["metallicRoughnessTexture"]["texCoord"],
                0u, mr_coord);
            if (mat_err) return false;

            if (bc_coord != 0 || mr_coord != 0) {
                loader.recordError("Multiple UVs not supported.");
                return false;
            }

            math::Vector4 base_color;
            mat_err = jsonGetOr(loader, pbr["baseColorFactor"],
                math::Vector4::one(), base_color);
            if (mat_err) return false;

            float metallic;
            mat_err = jsonGetOr(loader, pbr["metallicFactor"], 1.f, metallic);
            if (mat_err) return false;

            float roughness;
            mat_err =
                jsonGetOr(loader, pbr["roughnessFactor"], 1.f, roughness);
            if (mat_err) return false;

            auto exts = material["extensions"];
            auto transmission_ext =
                exts["KHR_materials_transmission"];

            uint32_t transmission_idx;
            mat_err = jsonGetOr(
                loader, transmission_ext["transmissionTexture"]["index"],
                tex_missing, transmission_idx);
            if (mat_err) return false;

            float transmission_factor;
            mat_err = jsonGetOr(loader, transmission_ext["transmissionFactor"],
                                0.f, transmission_factor);
            if (mat_err) return false;

            auto specular_ext = exts["KHR_materials_specular"];

            math::Vector3 base_specular;
            mat_err = jsonGetOr(loader, specular_ext["specularColorFactor"],
                                math::Vector3::one(), base_specular);
            if (mat_err) return false;

            float specular_factor;
            mat_err = jsonGetOr(loader, specular_ext["specularFactor"],
                                1.f, specular_factor);
            if (mat_err) return false;

            uint32_t spec_idx;
            mat_err = jsonGetOr(
                loader, specular_ext["specularTexture"]["index"],
                tex_missing, spec_idx);
            if (mat_err) return false;

            uint32_t spec_color_idx;
            mat_err = jsonGetOr(
                loader, specular_ext["specularColorTexture"]["index"],
                tex_missing, spec_color_idx);
            if (mat_err) return false;

            if (spec_idx != spec_color_idx) {
                loader.recordError(
                    "Specular textures must be packed together");
                return false;
            }

            float ior;
            mat_err = jsonGetOr(
                loader, exts["KHR_materials_ior"]["ior"], 1.5f, ior);
            if (mat_err) return false;

            auto clearcoat_ext = exts["KHR_materials_clearcoat"];

            float clearcoat;
            mat_err = jsonGetOr(
                loader, clearcoat_ext["clearcoatFactor"], 0.f, clearcoat);
            if (mat_err) return false;

            float clearcoat_roughness;
            mat_err = jsonGetOr(
                loader, clearcoat_ext["clearcoatRoughnessFactor"],
                0.f, clearcoat_roughness);
            if (mat_err) return false;

            uint32_t clearcoat_idx;
            mat_err = jsonGetOr(
                loader, clearcoat_ext["clearcoatTexture"]["index"],
                tex_missing, clearcoat_idx);
            if (mat_err) return false;

            uint32_t clearcoat_roughness_idx;
            mat_err = jsonGetOr(
                loader, clearcoat_ext["clearcoatRoughnessTexture"]["index"],
                tex_missing, clearcoat_roughness_idx);
            if (mat_err) return false;

            uint32_t clearcoat_normal_idx;
            mat_err = jsonGetOr(
                loader, clearcoat_ext["clearcoatNormalTexture"]["index"],
                tex_missing, clearcoat_normal_idx);
            if (mat_err) return false;

            if (clearcoat_idx != clearcoat_roughness_idx) {
                loader.recordError(
                    "Clearcoat textures must be packed together");
                return false;
            }

            auto volume_ext = exts["KHR_materials_volume"];

            float thickness;
            mat_err = jsonGetOr(
                loader, volume_ext["thicknessFactor"], 0.f, thickness);
            if (mat_err) return false;

            bool thinwalled = thickness == 0.f;

            float attenuation_distance;
            mat_err = jsonGetOr(
                loader, volume_ext["attenuationDistance"],
                INFINITY, attenuation_distance);
            if (mat_err) return false;

            math::Vector3 attenuation_color;
            mat_err = jsonGetOr(
                loader, volume_ext["attenuationColor"],
                math::Vector3::one(), attenuation_color);
            if (mat_err) return false;

            auto aniso_ext = exts["KHR_materials_anisotropy"];

            float aniso_scale;
            mat_err = jsonGetOr(
                loader, aniso_ext["anisotropy"], 0.f, aniso_scale);
            if (mat_err) return false;

            math::Vector3 aniso_dir;
            mat_err = jsonGetOr(loader, aniso_ext["anisotropyDirection"],
                                math::Vector3 {1, 0, 0}, aniso_dir);
            if (mat_err) return false;

            uint32_t aniso_idx;
            mat_err = jsonGetOr(loader, aniso_ext["anisotropyTexture"],
                                tex_missing, aniso_idx);
            if (mat_err) return false;

            uint32_t aniso_rot_idx;
            mat_err = jsonGetOr(
                loader, aniso_ext["anisotropyDirectionTexture"],
                tex_missing, aniso_rot_idx);
            if (mat_err) return false;

            if (aniso_idx != aniso_rot_idx) {
                loader.recordError(
                    "Anisotropy textures must be packed together");
                return false;
            }

            uint32_t normal_idx;
            mat_err = jsonGetOr(
                loader, material["normalTexture"]["index"],
                tex_missing, normal_idx);
            if (mat_err) return false;

            math::Vector3 base_emittance;
            mat_err = jsonGetOr(
                loader, material["emissiveFactor"],
                math::Vector3::zero(), base_emittance);
            if (mat_err) return false;

            uint32_t emissive_idx;
            mat_err = jsonGetOr(
                loader, material["emissiveTexture"]["index"],
                tex_missing, emissive_idx);
            if (mat_err) return false;

            string_view material_name_view;
            string material_name;
            auto name_err = material["name"].get(material_name_view);
            if (name_err) {
                material_name = std::to_string(loader.materials.size());
            } else {
                material_name = material_name_view;
            }

            loader.materials.push_back(GLTFMaterial {
                std::move(material_name),
                base_color_idx,
                metallic_roughness_idx,
                spec_idx,
                normal_idx,
                emissive_idx,
                transmission_idx,
                clearcoat_idx,
                clearcoat_normal_idx,
                aniso_idx,
                base_color,
                transmission_factor,
                base_specular,
                specular_factor,
                metallic,
                roughness,
                ior,
                clearcoat,
                clearcoat_roughness,
                attenuation_color,
                attenuation_distance,
                aniso_scale,
                aniso_dir,
                base_emittance,
                thinwalled,
            });
        }
    }

    //cout << "materials" << endl;
    
    auto meshes = json_doc["meshes"].get_array();
    if (meshes.error()) {
        loader.recordJSONError(meshes.error());
        return false;
    }

    for (auto mesh : meshes.value_unsafe()) {
        auto gltf_prims = mesh["primitives"].get_array();

        if (gltf_prims.error()) {
            loader.recordJSONError(gltf_prims.error());
            return false;
        }

        uint32_t prim_offset = loader.prims.size();
        uint32_t num_prims = 0;
        for (auto prim : gltf_prims.value_unsafe()) {
            auto attrs = prim["attributes"];

            auto normal_idx = Optional<uint32_t>::none();
            auto uv_idx = Optional<uint32_t>::none();
            auto color_idx = Optional<uint32_t>::none();

            auto position_res = attrs["POSITION"].get_uint64();
            if (position_res.error()) {
                loader.recordJSONError(position_res.error());
                return false;
            }

            uint32_t position_idx = position_res.value_unsafe();

            uint64_t normal_res;
            auto normal_error = attrs["NORMAL"].get(normal_res);
            if (!normal_error) {
                normal_idx = uint32_t(normal_res);
            }

            uint64_t uv_res;
            auto uv_error = attrs["TEXCOORD_0"].get(uv_res);
            if (!uv_error) {
                uv_idx = uint32_t(uv_res);
            }

            uint64_t color_res;
            auto color_error = attrs["COLOR_0"].get(color_res);
            if (!color_error) {
                color_idx = uint32_t(color_res);
            }
            
            uint64_t material_idx;
            auto mat_error = prim["material"].get(material_idx);
            if (mat_error) {
                material_idx = 0;
            }

            uint64_t indices_idx;
            auto idx_error = prim["indices"].get(indices_idx);
            if (idx_error) {
                indices_idx = ~0u;
            }

            loader.prims.push_back({
                position_idx,
                normal_idx,
                uv_idx,
                color_idx,
                uint32_t(indices_idx),
                uint32_t(material_idx),
            });

            num_prims++;
        }

        string_view mesh_name_view;
        string mesh_name;
        auto name_err = mesh["name"].get(mesh_name_view);
        if (name_err) {
            mesh_name = std::to_string(loader.meshes.size());
        } else {
            mesh_name = mesh_name_view;
        }

        loader.meshes.push_back(GLTFMesh {
            std::move(mesh_name),
            prim_offset,
            num_prims,
        });
    }

    //cout << "meshes" << endl;
    
    auto nodes = json_doc["nodes"].get_array();
    if (nodes.error()) {
        loader.recordJSONError(nodes.error());
        return false;
    }

    for (auto node : nodes.value_unsafe()) {
        uint32_t child_node_offset = loader.childNodes.size();
        uint32_t num_children = 0;

        auto json_children = node["children"].get_array();

        if (!json_children.error()) {
            for (auto child : json_children.value_unsafe()) {
                auto child_idx = child.get_uint64();
                if (child_idx.error()) {
                    loader.recordJSONError(child_idx.error());
                    return false;
                }

                loader.childNodes.push_back(
                    uint32_t(child_idx.value_unsafe()));

                num_children++;
            }
        }

        uint64_t mesh_idx;
        auto mesh_error = node["mesh"].get(mesh_idx);
        if (mesh_error) {
            mesh_idx = loader.meshes.size();
        }

        math::Mat3x4 txfm = math::Mat3x4::identity();

        auto matrix = node["matrix"].get_array();
        if (!matrix.error()) {
            CountT cur_col = 0;
            CountT cur_row = 0;
            for (auto mat_elem : matrix) {
                if (cur_col == 4) {
                    loader.recordError("Invalid matrix transform");
                }

                auto v_res = mat_elem.get_double();
                if (v_res.error()) {
                    loader.recordJSONError(v_res.error());
                    return false;
                }

                float v = float(v_res.value_unsafe());

                if (cur_row < 3) {
                    txfm.cols[cur_col][cur_row] = v;
                } else {
                    if (cur_col < 3) {
                        if (v != 0) {
                            loader.recordError("Invalid matrix transform");
                            return false;
                        }
                    } else {
                        if (v != 1) {
                            loader.recordError("Invalid matrix transform");
                            return false;
                        }
                    }
                }

                cur_row++;

                if (cur_row == 4) {
                    cur_col++;
                    cur_row = 0;
                }
            }
        } else {
            auto translation = math::Vector3::zero();

            auto translation_arr = node["translation"].get_array();
            if (!translation_arr.error()) {
                CountT component_idx = 0;
                for (auto vec_elem : translation_arr.value_unsafe()) {
                    auto v = vec_elem.get_double();

                    if (v.error()) {
                        loader.recordJSONError(v.error());
                        return false;
                    }

                    if (component_idx < 3) {
                        translation[component_idx] = v.value_unsafe();
                    }

                    component_idx++;
                }

                if (component_idx != 3) {
                    loader.recordError(
                        "Node translation with wrong number of components");
                    return false;
                }
            }

            math::Quat rotation { 1, 0, 0, 0 };
            auto quat_arr = node["rotation"].get_array();
            if (!quat_arr.error()) {
                CountT component_idx = 0;
                for (auto vec_elem : quat_arr.value_unsafe()) {
                    auto v = vec_elem.get_double();

                    if (v.error()) {
                        loader.recordJSONError(v.error());
                        return false;
                    }

                    float f = v.value_unsafe();

                    if (component_idx == 0) {
                        rotation.x = f;
                    } else if (component_idx == 1) {
                        rotation.y = f;
                    } else if (component_idx == 2) {
                        rotation.z = f;
                    } else if (component_idx == 3) {
                        rotation.w = f;
                    }

                    component_idx++;
                }

                if (component_idx != 4) {
                    loader.recordError(
                        "Node rotation with wrong number of components");
                    return false;
                }
            }


            math::Diag3x3 scale { 1, 1, 1 };
            auto scale_arr = node["scale"].get_array();
            if (!scale_arr.error()) {
                CountT component_idx = 0;
                for (auto vec_elem : scale_arr.value_unsafe()) {
                    auto v = vec_elem.get_double();

                    if (v.error()) {
                        loader.recordJSONError(v.error());
                        return false;
                    }

                    float f = v.value_unsafe();

                    if (component_idx == 0) {
                        scale.d0 = f;
                    } else if (component_idx == 1) {
                        scale.d1 = f;
                    } else if (component_idx == 2) {
                        scale.d2 = f;
                    }

                    component_idx++;
                }

                if (component_idx != 3) {
                    loader.recordError(
                        "Node scale with wrong number of components");
                    return false;
                }
            }

            txfm = math::Mat3x4::fromTRS(translation, rotation, scale);
        }

        loader.nodes.push_back(GLTFNode {
            child_node_offset,
            num_children,
            static_cast<uint32_t>(mesh_idx),
            txfm,
        });
    }

    //cout << "nodes" << endl;

    auto scenes = json_doc["scenes"].get_array();
    if (scenes.error()) {
        loader.recordJSONError(scenes.error());
        return false;
    }

    CountT scene_idx = 0;
    for (auto scene : scenes.value_unsafe()) {
        if (scene_idx != 0) {
            loader.recordError("Multiscene files not supported");
            return false;
        }

        auto scene_nodes = scene["nodes"].get_array();
        if (scene_nodes.error()) {
            loader.recordJSONError(scene_nodes.error());
            return false;
        }

        for (auto node : scene_nodes.value_unsafe()) {
            auto node_idx = node.get_uint64();

            if (node_idx.error()) {
                loader.recordJSONError(node_idx.error());
                return false;
            }

            loader.rootNodes.push_back(node_idx.value_unsafe());
        }

        scene_idx++;
    }

    return true;
}

template <typename T>
static Optional<GLTFStridedSpan<T>> getGLTFBufferView(
    const LoaderData &loader,
    uint32_t view_idx,
    uint32_t start_offset = 0,
    uint32_t num_elems = 0)
{
    const GLTFBufferView &view = loader.bufferViews[view_idx];
    const GLTFBuffer &buffer = loader.buffers[view.bufferIdx];

    if (buffer.dataPtr == nullptr) {
        loader.recordError(
            "GLTF loading failed: external references not supported");
        return Optional<GLTFStridedSpan<T>>::none();
    }

    size_t total_offset = start_offset + view.offset;
    const uint8_t *start_ptr = buffer.dataPtr + total_offset;

    uint32_t stride = view.stride;
    if (stride == 0) {
        stride = sizeof(T);
    }

    if (num_elems == 0) {
        num_elems = view.numBytes / stride;
    }

    return GLTFStridedSpan<T>(start_ptr, num_elems, stride);
}

template <typename T>
static Optional<GLTFStridedSpan<T>> getGLTFAccessorView(
    const LoaderData &loader,
    uint32_t accessor_idx)
{
    const GLTFAccessor &accessor = loader.accessors[accessor_idx];

    return getGLTFBufferView<T>(loader, accessor.viewIdx, accessor.offset,
                                accessor.numElems);
}

// GLTF Mesh = Madrona Object, Primitive = Madrona Mesh
static bool gltfParseMesh(
    CountT mesh_idx,
    const LoaderData &loader, 
    ImportedAssets &imported)
{
    const GLTFMesh &gltf_mesh = loader.meshes[mesh_idx];

    DynArray<SourceMesh> meshes(1);
    for (CountT prim_offset = 0; prim_offset < (CountT)gltf_mesh.numPrims;
         prim_offset++) {
        CountT prim_idx = prim_offset + gltf_mesh.primOffset;
        const GLTFPrimitive &prim = loader.prims[prim_idx];

        auto position_accessor = getGLTFAccessorView<const math::Vector3>(
            loader, prim.positionIdx);

        if (!position_accessor.has_value()) {
            return false;
        }

        auto normal_accessor =
            Optional<GLTFStridedSpan<const math::Vector3>>::none();

        if (prim.normalIdx.has_value()) {
            normal_accessor = getGLTFAccessorView<const math::Vector3>(
                loader, *prim.normalIdx);

            if (!normal_accessor.has_value()) {
                return false;
            }
        }

        auto uv_accessor =
            Optional<GLTFStridedSpan<const math::Vector2>>::none();

        if (prim.uvIdx.has_value()) {
            uv_accessor = getGLTFAccessorView<const math::Vector2>(
                loader, *prim.uvIdx);

            if (!uv_accessor.has_value()) {
                return false;
            }
        }

        uint32_t max_idx = 0;

        DynArray<uint32_t> indices(0);
        if (prim.indicesIdx != ~0u) {
            auto index_type = loader.accessors[prim.indicesIdx].type;

            if (index_type == GLTFComponentType::UINT32) {
                auto idx_accessor = getGLTFAccessorView<const uint32_t>(
                    loader, prim.indicesIdx);
                if (!idx_accessor.has_value()) {
                    return false;
                }

                indices.reserve(idx_accessor->size());

                for (uint32_t idx : *idx_accessor) {
                    if (idx > max_idx) {
                        max_idx = idx;
                    }

                    indices.push_back(idx);
                }
            } else if (index_type == GLTFComponentType::UINT16) {
                auto idx_accessor = getGLTFAccessorView<const uint16_t>(
                    loader, prim.indicesIdx);
                if (!idx_accessor.has_value()) {
                    return false;
                }

                indices.reserve(idx_accessor->size());

                for (uint16_t idx : *idx_accessor) {
                    if (idx > max_idx) {
                        max_idx = idx;
                    }

                    indices.push_back(idx);
                }
            } else {
                loader.recordError(
                    "GLTF loading failed: unsupported index type");
                return false;
            }
        } else {
            indices.reserve(position_accessor->size());

            for (CountT i = 0; i < (CountT)position_accessor->size(); i++) {
                indices.push_back(uint32_t(i));
            }

            max_idx = position_accessor->size() - 1;
        }

        uint32_t num_faces = indices.size() / 3;
        if (num_faces * 3 != indices.size()) {
            loader.recordError("Non-triangular GLTF not supported");
            return false;
        }

        uint32_t num_vertices = max_idx + 1;

        DynArray<math::Vector3> positions(num_vertices);
        auto normals = Optional<DynArray<math::Vector3>>::none();
        auto uvs = Optional<DynArray<math::Vector2>>::none();

        if (normal_accessor.has_value()) {
            if (normal_accessor->size() != position_accessor->size()) {
                loader.recordError("Fewer normals than positions in mesh %d",
                                   mesh_idx);
                return false;
            }
            normals.emplace(num_vertices);
        }

        if (uv_accessor.has_value()) {
            if (uv_accessor->size() != position_accessor->size()) {
                loader.recordError("Fewer UVs than positions in mesh %d",
                                   mesh_idx);
                return false;
            }
            uvs.emplace(num_vertices);
        }

        for (uint32_t vert_idx = 0; vert_idx < num_vertices; vert_idx++) {
            math::Vector3 pos = (*position_accessor)[vert_idx];
            if (isnan(pos.x) || isinf(pos.x)) {
                pos.x = 0;
            }

            if (isnan(pos.y) || isinf(pos.y)) {
                pos.y = 0;
            }

            if (isnan(pos.z) || isinf(pos.z)) {
                pos.z = 0;
            }
            positions.push_back(pos);

            if (normal_accessor.has_value()) {
                math::Vector3 normal = (*normal_accessor)[vert_idx];

                if (isnan(normal.x) || isinf(normal.x)) {
                    normal.x = 0;
                }

                if (isnan(normal.y) || isinf(normal.y)) {
                    normal.y = 0;
                }

                if (isnan(normal.z) || isinf(normal.z)) {
                    normal.z = 0;
                }

                normals->push_back(normal);
            }

            if (uv_accessor.has_value()) {
                math::Vector2 uv = (*uv_accessor)[vert_idx];

                if (isnan(uv.x) || isinf(uv.x)) {
                    uv.x = 0;
                }

                if (isnan(uv.y) || isinf(uv.y)) {
                    uv.y = 0;
                }

                uvs->push_back(uv);
            }
        }

        math::Vector3 *position_ptr = positions.data();
        imported.geoData.positionArrays.emplace_back(std::move(positions));

        math::Vector3 *normal_ptr = nullptr;
        if (normals.has_value()) {
            normal_ptr = normals->data();
            imported.geoData.normalArrays.emplace_back(std::move(*normals));
        }

        math::Vector2 *uv_ptr = nullptr;
        if (uvs.has_value()) {
            uv_ptr = uvs->data();
            imported.geoData.uvArrays.emplace_back(std::move(*uvs));
        }

        uint32_t *idx_ptr = indices.data();
        imported.geoData.indexArrays.emplace_back(std::move(indices));

        meshes.push_back(SourceMesh {
            .positions = position_ptr,
            .normals = normal_ptr,
            .tangentAndSigns = nullptr,
            .uvs = uv_ptr,
            .indices = idx_ptr,
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = num_vertices,
            .numFaces = num_faces,
            .materialIDX = 0, // FIXME
        });
    }

    imported.objects.push_back({
        .meshes = { meshes.data(), meshes.size() },
    });

    imported.geoData.meshArrays.emplace_back(std::move(meshes));

    return true;
}

static bool gltfParseInstances(const LoaderData &loader,
                               ImportedAssets &imported,
                               CountT base_obj_idx)
{
    DynArray<std::pair<uint32_t, math::Mat3x4>> node_stack(
        loader.rootNodes.size());
    for (uint32_t root_node : loader.rootNodes) {
        node_stack.emplace_back(root_node, math::Mat3x4::identity());
    }

    while (node_stack.size() != 0) {
        auto [node_idx, parent_txfm] = node_stack.back();
        node_stack.pop_back();

        const GLTFNode &cur_node = loader.nodes[node_idx];
        math::Mat3x4 cur_txfm = parent_txfm.compose(cur_node.transform);

        for (uint32_t child_offset = 0; child_offset < cur_node.numChildren;
             child_offset++) {
            uint32_t child_idx = child_offset + cur_node.childOffset;
            uint32_t child_node_idx = loader.childNodes[child_idx];

            node_stack.emplace_back(child_node_idx, cur_txfm);
        }

        math::Vector3 translation;
        math::Quat rotation;
        math::Diag3x3 scale;

        cur_txfm.decompose(&translation, &rotation, &scale);

        if (cur_node.meshIdx < loader.meshes.size()) {
            imported.instances.push_back(SourceInstance {
                translation,
                rotation,
                scale,
                uint32_t(base_obj_idx + cur_node.meshIdx),
            });
        }
    }

    return true;
}

static bool gltfImportAssets(LoaderData &loader,
                             ImportedAssets &imported,
                             bool merge_and_flatten)
{
    CountT new_mesh_arrays_start = imported.geoData.meshArrays.size();
    CountT new_vert_arrays_start = imported.geoData.positionArrays.size();
    CountT new_normal_arrays_start = imported.geoData.normalArrays.size();
    CountT new_objects_start = imported.objects.size();
    CountT new_instances_start = imported.instances.size();

    for (CountT mesh_idx = 0; mesh_idx < loader.meshes.size();
         mesh_idx++) {
        bool mesh_valid = gltfParseMesh(mesh_idx, loader, imported);
        if (!mesh_valid) {
            return false;
        }
    }

    bool instances_valid =
        gltfParseInstances(loader, imported, new_objects_start);

    if (!instances_valid) {
        return false;
    }

    if (!merge_and_flatten) {
        return true;
    }

    CountT total_new_vertices = 0;
    CountT total_new_normals = 0;
    CountT total_new_tangents = 0;
    CountT total_new_meshes = 0;
    for (CountT inst_idx = new_instances_start;
         inst_idx < imported.instances.size(); inst_idx++) {
        const SourceInstance &inst = imported.instances[inst_idx];
        const SourceObject &src_obj = imported.objects[inst.objIDX];
        for (const SourceMesh &src_mesh : src_obj.meshes) {
            total_new_vertices += src_mesh.numVertices;

            if (src_mesh.normals) {
                total_new_normals += src_mesh.numVertices;
            }

            if (src_mesh.tangentAndSigns) {
                total_new_tangents += src_mesh.numVertices;
            }
        }

        total_new_meshes += src_obj.meshes.size();
    }

    DynArray<SourceMesh> merged_meshes(total_new_meshes);
    DynArray<math::Vector3> new_positions_arr(total_new_vertices);
    DynArray<math::Vector3> new_normals_arr(total_new_normals);
    DynArray<math::Vector4> new_tangentsigns_arr(total_new_tangents);

    for (CountT inst_idx = new_instances_start;
         inst_idx < imported.instances.size(); inst_idx++) {
        const SourceInstance &inst = imported.instances[inst_idx];
        const SourceObject &src_obj = imported.objects[inst.objIDX];

        math::Vector3 *new_mesh_positions_ptr =
            new_positions_arr.data() + new_positions_arr.size();
        math::Vector3 *new_mesh_normals_ptr =
            new_normals_arr.data() + new_normals_arr.size();

        math::Vector4 *new_mesh_tangents_ptr =
            new_tangentsigns_arr.data() + new_tangentsigns_arr.size();

        for (const SourceMesh &src_mesh : src_obj.meshes) {
            for (CountT i = 0; i < src_mesh.numVertices; i++) {
                math::Vector3 orig_pos = src_mesh.positions[i];

                math::Vector3 new_pos =
                    inst.rotation.rotateVec(inst.scale * orig_pos) +
                    inst.translation;
                new_positions_arr.push_back(new_pos);

                if (src_mesh.normals) {
                    math::Vector3 orig_normal = src_mesh.normals[i];
                    math::Vector3 new_normal =
                        inst.rotation.rotateVec(inst.scale.inv() * orig_normal);
                    new_normals_arr.push_back(new_normal);
                }

                if (src_mesh.tangentAndSigns) {
                    math::Vector3 orig_tangent = src_mesh.tangentAndSigns[i].xyz();
                    math::Vector3 new_tangent = inst.rotation.rotateVec(
                        inst.scale * orig_tangent) + inst.translation;
                    new_tangentsigns_arr.push_back(math::Vector4 {
                        new_tangent.x,
                        new_tangent.y,
                        new_tangent.z,
                        src_mesh.tangentAndSigns[i].w,
                    });
                }
            }

            merged_meshes.push_back(SourceMesh {
                .positions = new_mesh_positions_ptr,
                .normals = src_mesh.normals ?
                    new_mesh_normals_ptr : nullptr,
                .tangentAndSigns = src_mesh.tangentAndSigns ?
                    new_mesh_tangents_ptr : nullptr,
                .uvs = src_mesh.uvs,
                .indices = src_mesh.indices,
                .faceCounts = src_mesh.faceCounts,
                .numVertices = src_mesh.numVertices,
                .numFaces = src_mesh.numFaces,
                .materialIDX = src_mesh.materialIDX,
            });
        }
    }

    imported.geoData.meshArrays.resize(new_mesh_arrays_start, [](auto *) {});
    imported.geoData.positionArrays.resize(new_vert_arrays_start,
                                           [](auto *) {});
    imported.geoData.normalArrays.resize(new_normal_arrays_start,
                                         [](auto *) {});
    imported.objects.resize(new_objects_start,
                            [](auto *) {});
    imported.instances.resize(new_instances_start,
                              [](auto *) {});

    imported.objects.push_back({
        .meshes = Span<SourceMesh>(
            merged_meshes.data(), merged_meshes.size()),
    });
    
    imported.instances.push_back(SourceInstance {
        .translation = math::Vector3::zero(),
        .rotation = math::Quat { 1, 0, 0, 0 },
        .scale = math::Diag3x3::uniform(1.f),
        .objIDX = uint32_t(new_objects_start),
    });

    imported.geoData.meshArrays.emplace_back(std::move(merged_meshes));
    imported.geoData.positionArrays.emplace_back(std::move(new_positions_arr));
    imported.geoData.normalArrays.emplace_back(std::move(new_normals_arr));

    return true;
}

GLTFLoader::Impl::Impl(Span<char> err_buf)
    : errBuf(err_buf),
      jsonBuf(0),
      jsonParser(),
      curFileName(nullptr),
      sceneDirectory(),
      sceneName(),
      internalData(0),
      buffers(0),
      bufferViews(0),
      accessors(0),
      images(0),
      textures(0),
      materials(0),
      prims(0),
      meshes(0),
      childNodes(0),
      nodes(0),
      rootNodes(0)
{}

void GLTFLoader::Impl::recordError(const char *fmt, ...) const
{
    if (errBuf.data() == nullptr) {
        return;
    }

    int prefix_chars_written = snprintf(errBuf.data(), errBuf.size(),
        "Invalid GLTF File %s: ", curFileName);

    if (prefix_chars_written < errBuf.size()) {
        va_list args;
        va_start(args, fmt);

        size_t remaining = errBuf.size() - prefix_chars_written;

        vsnprintf(errBuf.data() + prefix_chars_written, remaining,
                  fmt, args);
    }
}

void GLTFLoader::Impl::recordJSONError(error_code err) const
{
    if (errBuf.data() == nullptr) {
        return;
    }

    snprintf(errBuf.data(), errBuf.size(),
        "Invalid GLTF File %s\nJSON Error: %s", curFileName,
        error_message(err));
}

GLTFLoader::GLTFLoader(Span<char> err_buf)
    : impl_(new Impl(err_buf))
{}

GLTFLoader::~GLTFLoader() {}

bool GLTFLoader::load(const char *path, 
                      ImportedAssets &imported_assets,
                      bool merge_and_flatten)
{
    bool json_parsed = gltfLoad(path, *impl_);
    if (!json_parsed) {
        return false;
    }

    bool import_success = gltfImportAssets(*impl_, imported_assets,
                                           merge_and_flatten);
    if (!import_success) {
        return false;
    }

    // Clear tmp buffers
    impl_->internalData.clear();
    impl_->buffers.clear();
    impl_->bufferViews.clear();
    impl_->accessors.clear();
    impl_->images.clear();
    impl_->textures.clear();
    impl_->materials.clear();
    impl_->prims.clear();
    impl_->meshes.clear();
    impl_->childNodes.clear();
    impl_->nodes.clear();
    impl_->rootNodes.clear();

    return true;
}

}

// Old code from rlpbr
#if 0
static inline void dumpGLTFTexture(const GLTFScene &scene,
                                   const GLTFImage &img,
                                   string_view texture_name,
                                   texutil::TextureType texture_type,
                                   const TextureCallback &texture_cb)
{
    const GLTFBufferView &buffer_view = scene.bufferViews[img.viewIdx];
    if (buffer_view.stride > 1) {
        cerr << "GLTF import: cannot dump strided texture" << endl;
        abort();
    }

    const uint8_t *texture_ptr = scene.internalData.data() + buffer_view.offset;

    texture_cb(texture_name, texture_type,
               texture_ptr, buffer_view.numBytes);
}

template <typename MaterialType>
vector<MaterialType> gltfParseMaterials(const GLTFScene &scene,
    const TextureCallback &texture_cb)
{
    vector<MaterialType> materials;
    materials.reserve(scene.materials.size());

    unordered_set<uint32_t> texture_tracker;

    auto extractTex = [&](uint32_t tex_idx, texutil::TextureType tex_type) {
        string tex_name = "";
        if (tex_idx < scene.textures.size()) {
            const GLTFImage &img =
                scene.images[scene.textures[tex_idx].sourceIdx];

            if (img.type == GLTFImageType::EXTERNAL) {
                tex_name = img.filePath;
            } else {
                tex_name = scene.sceneName + "_" + to_string(tex_idx);

                auto inserted = texture_tracker.emplace(tex_idx);
                if (inserted.second) {
                    dumpGLTFTexture(scene, img, tex_name, tex_type,
                                    texture_cb);
                }
            }
        }

        return tex_name;
    };

    for (const auto &gltf_mat : scene.materials) {
        float alpha_transparency = gltf_mat.baseColor.a;

        // Hack: use alpha as proper transparency for assets with
        // hardcoded alpha in the baseColorFactor
        float transmission = gltf_mat.transmissionFactor;
        if (transmission == 0.f) {
            transmission = 1.f - alpha_transparency;
        }

        glm::vec3 base_color(gltf_mat.baseColor.r,
                             gltf_mat.baseColor.g,
                             gltf_mat.baseColor.b);

        // FIXME
        float aniso_rotation = atan2(gltf_mat.anisoDir.y, gltf_mat.anisoDir.x);

        materials.push_back({
            gltf_mat.name,
            extractTex(gltf_mat.baseColorIdx,
                       texutil::TextureType::FourChannelSRGB),
            extractTex(gltf_mat.metallicRoughnessIdx,
                       texutil::TextureType::TwoChannelLinearGB),
            extractTex(gltf_mat.specularIdx,
                       texutil::TextureType::FourChannelSRGB),
            extractTex(gltf_mat.normalIdx,
                       texutil::TextureType::NormalMap),
            extractTex(gltf_mat.emittanceIdx,
                       // FIXME: should be HDR
                       texutil::TextureType::FourChannelSRGB),
            extractTex(gltf_mat.transmissionIdx,
                       texutil::TextureType::SingleChannelLinearR),
            extractTex(gltf_mat.clearcoatIdx,
                       texutil::TextureType::TwoChannelLinearRG),
            extractTex(gltf_mat.anisoIdx,
                       texutil::TextureType::TwoChannelLinearRG),
            base_color,
            transmission,
            gltf_mat.baseSpecular,
            gltf_mat.specularFactor,
            gltf_mat.metallic,
            gltf_mat.roughness,
            gltf_mat.ior,
            gltf_mat.clearcoat,
            gltf_mat.clearcoatRoughness,
            gltf_mat.attenuationColor,
            gltf_mat.attenuationDistance,
            gltf_mat.anisoScale,
            aniso_rotation,
            gltf_mat.baseEmittance,
            gltf_mat.thinwalled,
        });
    }

    return materials;
}

template <typename T, typename = int>
struct HasPosition : std::false_type { };
template <typename T>
struct HasPosition <T, decltype((void) T::position, 0)> : std::true_type { };

template <typename T, typename = int>
struct HasNormal : std::false_type { };
template <typename T>
struct HasNormal <T, decltype((void) T::normal, 0)> : std::true_type { };

template <typename T, typename = int>
struct HasUV : std::false_type { };
template <typename T>
struct HasUV <T, decltype((void) T::uv, 0)> : std::true_type { };

template <typename T, typename = int>
struct HasColor : std::false_type { };
template <typename T>
struct HasColor <T, decltype((void) T::color, 0)> : std::true_type { };

template <typename VertexType>
static vector<Mesh<VertexType>> gltfParseMesh(
    const GLTFScene &scene,
    uint32_t mesh_idx)
{
    const GLTFMesh &gltf_mesh = scene.meshes[mesh_idx];

    vector<Mesh<VertexType>> meshes;

    for (const GLTFPrimitive &prim : gltf_mesh.primitives) {
        vector<VertexType> vertices;
        vector<uint32_t> indices;

        optional<GLTFStridedSpan<const glm::vec3>> position_accessor;
        optional<GLTFStridedSpan<const glm::vec3>> normal_accessor;
        optional<GLTFStridedSpan<const glm::vec2>> uv_accessor;
        optional<GLTFStridedSpan<const glm::u8vec3>> color_accessor;

        constexpr bool has_position = HasPosition<VertexType>::value;
        constexpr bool has_normal = HasNormal<VertexType>::value;
        constexpr bool has_uv = HasUV<VertexType>::value;
        constexpr bool has_color = HasColor<VertexType>::value;

        if constexpr (has_position) {
            position_accessor = getGLTFAccessorView<const glm::vec3>(
                scene, prim.positionIdx.value());
        }

        if constexpr (has_normal) {
            if (prim.normalIdx.has_value()) {
                normal_accessor = getGLTFAccessorView<const glm::vec3>(
                    scene, prim.normalIdx.value());
            }
        }

        if constexpr (has_uv) {
            if (prim.uvIdx.has_value()) {
                uv_accessor = getGLTFAccessorView<const glm::vec2>(scene,
                    prim.uvIdx.value());
            }
        }

        if constexpr (has_color) {
            if (prim.colorIdx.has_value()) {
                color_accessor = getGLTFAccessorView<const glm::u8vec3>(
                    scene, prim.colorIdx.value());
            }
        }

        uint32_t max_idx = 0;

        if (prim.indicesIdx != ~0u) {
            auto index_type = scene.accessors[prim.indicesIdx].type;

            if (index_type == GLTFComponentType::UINT32) {
                auto idx_accessor =
                    getGLTFAccessorView<const uint32_t>(scene, prim.indicesIdx);
                indices.reserve(idx_accessor.size());

                for (uint32_t idx : idx_accessor) {
                    if (idx > max_idx) {
                        max_idx = idx;
                    }

                    indices.push_back(idx);
                }
            } else if (index_type == GLTFComponentType::UINT16) {
                auto idx_accessor =
                    getGLTFAccessorView<const uint16_t>(scene, prim.indicesIdx);
                indices.reserve(idx_accessor.size());

                for (uint16_t idx : idx_accessor) {
                    if (idx > max_idx) {
                        max_idx = idx;
                    }

                    indices.push_back(idx);
                }
            } else {
                cerr << "GLTF loading failed: unsupported index type"
                          << endl;
                abort();
            }
        } else {
            indices.reserve(position_accessor->size());

            for (int i = 0; i < (int)position_accessor->size(); i++) {
                indices.push_back(i);
            }

            max_idx = position_accessor->size() - 1;
        }

        max_idx = min(uint32_t(position_accessor->size()), max_idx);

        vertices.reserve(max_idx + 1);
        for (uint32_t vert_idx = 0; vert_idx <= max_idx; vert_idx++) {
            VertexType vert {};

            if constexpr (has_position) {
                vert.position = (*position_accessor)[vert_idx];
                if (isnan(vert.position.x) || isinf(vert.position.x)) {
                    vert.position.x = 0;
                }

                if (isnan(vert.position.y) || isinf(vert.position.y)) {
                    vert.position.y = 0;
                }

                if (isnan(vert.position.z) || isinf(vert.position.z)) {
                    vert.position.z = 0;
                }
            }

            if constexpr (has_normal) {
                if (normal_accessor.has_value()) {
                    vert.normal = (*normal_accessor)[vert_idx];
                }

                if (isnan(vert.normal.x) || isinf(vert.normal.x)) {
                    vert.normal.x = 0;
                }

                if (isnan(vert.normal.y) || isinf(vert.normal.y)) {
                    vert.normal.y = 0;
                }

                if (isnan(vert.normal.z) || isinf(vert.normal.z)) {
                    vert.normal.z = 0;
                }
            }

            if constexpr (has_uv) {
                if (uv_accessor.has_value()) {
                    vert.uv = (*uv_accessor)[vert_idx];
                }

                if (isnan(vert.uv.x) || isinf(vert.uv.x)) {
                    vert.uv.x = 0;
                }

                if (isnan(vert.uv.y) || isinf(vert.uv.y)) {
                    vert.uv.y = 0;
                }
            }

            if constexpr (has_color) {
                if (color_accessor.has_value()) {
                    vert.color = glm::u8vec4((*color_accessor)[vert_idx], 255);
                }
            }

            vertices.push_back(vert);
        }

        Mesh<VertexType> mesh {
            move(vertices),
            move(indices),
        };

        meshes.emplace_back(move(mesh));
    }

    return meshes;
}

template <typename MaterialType>
static std::vector<InstanceProperties> gltfParseInstances(
    const GLTFScene &scene,
    const glm::mat4 &coordinate_txfm,
    const vector<MaterialType> &materials)
{
    vector<pair<uint32_t, glm::mat4>> node_stack;
    for (uint32_t root_node : scene.rootNodes) {
        node_stack.emplace_back(root_node, coordinate_txfm);
    }

    vector<InstanceProperties> instances;
    while (!node_stack.empty()) {
        auto [node_idx, parent_txfm] = node_stack.back();
        node_stack.pop_back();

        const GLTFNode &cur_node = scene.nodes[node_idx];
        glm::mat4 cur_txfm = parent_txfm * cur_node.transform;

        for (const uint32_t child_idx : cur_node.children) {
            node_stack.emplace_back(child_idx, cur_txfm);
        }

        if (cur_node.meshIdx < scene.meshes.size()) {
            // Decompose transform
            glm::vec3 position(cur_txfm[3]);

            glm::vec3 scale(glm::length(cur_txfm[0]),
                            glm::length(cur_txfm[1]),
                            glm::length(cur_txfm[2]));

            if (glm::dot(glm::cross(glm::vec3(cur_txfm[0]),
                                    glm::vec3(cur_txfm[1])),
                         glm::vec3(cur_txfm[2])) < 0.f) {
                scale.x *= -1.f;
            }

            glm::vec3 v1 = glm::normalize(cur_txfm[0] / scale.x);
            glm::vec3 v2 = cur_txfm[1] / scale.y;
            glm::vec3 v3 = cur_txfm[2] / scale.z;

            v2 = glm::normalize(v2 - dot(v2, v1) * v1);
            v3 = v3 - glm::dot(v3, v1) * v1;
            v3 -= glm::dot(v3, v2) * v2;
            v3 = glm::normalize(v3);

            glm::mat3 rot(v1, v2, v3);

            glm::quat rot_quat = glm::quat_cast(rot);

            bool is_transparent = false;
            vector<uint32_t> instance_materials;
            for (const auto &prim : scene.meshes[cur_node.meshIdx].primitives) {
                instance_materials.push_back(prim.materialIdx);

                if (materials[prim.materialIdx].baseTransmission > 0.f) {
                    is_transparent = true;
                }
            }

            instances.push_back({
                std::to_string(instances.size()),
                cur_node.meshIdx,
                move(instance_materials),
                position,
                rot_quat,
                scale,
                false,
                is_transparent,
            });
        }
    }

    return instances;
}

template <typename VertexType, typename MaterialType>
SceneDescription<VertexType, MaterialType> parseGLTF(
    std::filesystem::path scene_path, const glm::mat4 &base_txfm,
    const TextureCallback &texture_cb)
{
    auto raw_scene = gltfLoad(scene_path);

    vector<MaterialType> materials =
        gltfParseMaterials<MaterialType>(raw_scene, texture_cb);

    vector<Object<VertexType>> geometry;

    for (uint32_t mesh_idx = 0; mesh_idx < raw_scene.meshes.size();
         mesh_idx++) {
        vector<Mesh<VertexType>> meshes =
            gltfParseMesh<VertexType>(raw_scene, mesh_idx);

        Object<VertexType> obj {
            raw_scene.meshes[mesh_idx].name,
            move(meshes),
        };

        geometry.emplace_back(move(obj));
    }

    vector<InstanceProperties> instances =
        gltfParseInstances(raw_scene, base_txfm, materials);

    return SceneDescription<VertexType, MaterialType> {
        move(geometry),
        move(materials),
        move(instances),
        {},
    };
}

static MergedSourceObject loadAndParseGLTF(
    std::filesystem::path gltf_path,
    const madrona::math::Mat3x4 &base_txfm_raw)
{
    GLTFScene raw_scene = gltfLoad(gltf_path);

    vector<vector<ParsedMesh>> geometry;
    for (size_t mesh_idx = 0; mesh_idx < raw_scene.meshes.size();
         mesh_idx++) {
        vector<ParsedMesh> meshes = gltfParseMesh(raw_scene, mesh_idx);
        geometry.emplace_back(move(meshes));
    }

    glm::mat4 base_txfm(
        glm::vec4(
            base_txfm_raw.cols[0].x,
            base_txfm_raw.cols[0].y,
            base_txfm_raw.cols[0].z,
            0),
        glm::vec4(
            base_txfm_raw.cols[1].x,
            base_txfm_raw.cols[1].y,
            base_txfm_raw.cols[1].z,
            0),
        glm::vec4(
            base_txfm_raw.cols[2].x,
            base_txfm_raw.cols[2].y,
            base_txfm_raw.cols[2].z,
            0),
        glm::vec4(
            base_txfm_raw.cols[3].x,
            base_txfm_raw.cols[3].y,
            base_txfm_raw.cols[3].z,
            1));

    auto instances = gltfParseInstances(raw_scene, base_txfm);

    MergedSourceObject merged;

    for (const auto &inst : instances) {
        const auto &src_obj_meshes = geometry[inst.objectIndex];
        for (const ParsedMesh &src_mesh : src_obj_meshes) {
            std::vector<math::Vector3> new_positions;
            std::vector<math::Vector3> new_normals;
            std::vector<math::Vector2> new_uvs;
            new_positions.reserve(src_mesh.vertices.size());
            new_normals.reserve(src_mesh.vertices.size());
            new_uvs.reserve(src_mesh.vertices.size());

            // copy
            std::vector<uint32_t> new_indices(src_mesh.indices);

            for (const auto &v : src_mesh.vertices) {
                glm::vec4 pos { v.position.x, v.position.y, v.position.z, 1.f };
                glm::vec4 txfm_pos = inst.txfm * pos;

                glm::vec3 normal { v.normal.x, v.normal.y, v.normal.z };
                glm::vec3 txfm_normal =
                    glm::normalize(inst.normalTxfm * normal);
                new_positions.push_back({txfm_pos.x, txfm_pos.y, txfm_pos.z });
                new_normals.push_back({txfm_normal.x, txfm_normal.y,
                    txfm_normal.z });
                new_uvs.push_back({ v.uv.x, v.uv.y });
            }

            merged.positions.emplace_back(std::move(new_positions));
            merged.normals.emplace_back(std::move(new_normals));
            merged.uvs.emplace_back(std::move(new_uvs));
            merged.indices.emplace_back(std::move(new_indices));

            imp::SourceMesh txfm_mesh {
                merged.positions.back().data(),
                merged.normals.back().data(),
                nullptr,
                merged.uvs.back().data(),
                merged.indices.back().data(),
                nullptr,
                uint32_t(src_mesh.vertices.size()),
                uint32_t(src_mesh.indices.size() / 3),
            };

            merged.meshes.push_back(txfm_mesh);
        }
    }

    return merged;
}


#endif
