#pragma once
#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/dyn_array.hpp>

#include <array>

// All the below classes are virtual because nanobind
// uses RTTI to match types across modules which only works with virtual types

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona::py {

#ifdef MADRONA_CUDA_SUPPORT
class CudaSync final {
public:
    CudaSync(cudaExternalSemaphore_t sema);
    void wait(uint64_t strm);

private:
#ifdef MADRONA_LINUX
    // These classes have to be virtual on linux so a unique typeinfo
    // is emitted. Otherwise every user of this class gets a weak symbol
    // reference and nanobind can't map the types correctly
    virtual void key_();
#endif

    cudaExternalSemaphore_t sema_;
};
#endif

// Need to wrap the actual enum class because macos
// RTTI for enum classes isn't consistent across libraries
class PyExecMode final {
public:
    inline PyExecMode(ExecMode v)
        : v_(v)
    {}

    inline operator ExecMode() const
    {
        return v_;
    }

private:
#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    ExecMode v_;
};

class Tensor final {
public:
    static inline constexpr int64_t maxDimensions = 16;

    enum class ElementType {
        UInt8,
        Int8,
        Int16,
        Int32,
        Int64,
        Float16,
        Float32,
    };

    class Printer {
    public:
        Printer(const Printer &) = delete;
        Printer(Printer &&o);
        ~Printer();

        void print() const;

    private:
        inline Printer(void *dev_ptr,
                       void *print_ptr,
                       ElementType type,
                       int64_t num_items,
                       int64_t num_bytes_per_item);

        void *dev_ptr_;
        void *print_ptr_;
        ElementType type_;
        int64_t num_items_;
        int64_t num_bytes_per_item_;

    friend class Tensor;
    };

    Tensor(void *dev_ptr, ElementType type,
           Span<const int64_t> dimensions,
           Optional<int> gpu_id);

    Tensor(const Tensor &o);
    
    inline void * devicePtr() const { return dev_ptr_; }
    inline ElementType type() const { return type_; }
    inline bool isOnGPU() const { return gpu_id_ != -1; }
    inline int gpuID() const { return gpu_id_; }
    inline int64_t numDims() const { return num_dimensions_; }
    inline const int64_t *dims() const { return dimensions_.data(); }
    int64_t numBytesPerItem() const;

    Printer makePrinter() const;
private:
#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    void *dev_ptr_;
    ElementType type_;
    int gpu_id_;

    int64_t num_dimensions_;
    std::array<int64_t, maxDimensions> dimensions_;
};

class TrainInterface {
public:
    struct NamedTensor {
        const char *name;
        Tensor hdl;
    };


    TrainInterface(std::initializer_list<NamedTensor> obs,
                   Tensor actions,
                   Tensor rewards,
                   Tensor dones,
                   Tensor resets,
                   Optional<Tensor> policy_assignments,
                   std::initializer_list<NamedTensor> stats = {});
    TrainInterface(TrainInterface &&o);
    ~TrainInterface();

    Span<const NamedTensor> observations() const;
    Tensor actions() const;
    Tensor rewards() const;
    Tensor dones() const;
    Tensor resets() const;
    Optional<Tensor> policyAssignments() const;
    Span<const NamedTensor> stats() const;

private:
    struct Impl;

#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    std::unique_ptr<Impl> impl_;
};

}
