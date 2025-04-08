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

enum class TensorElementType {
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
};

struct TensorInterface {
    TensorElementType type;
    Span<const int64_t> dimensions;
};

class Tensor final {
public:
    static inline constexpr int64_t maxDimensions = 16;

    class Printer {
    public:
        Printer(const Printer &) = delete;
        Printer(Printer &&o);
        ~Printer();

        void print(int64_t flatten_dim = 0) const;

    private:
        int64_t printInnerDims(void *print_ptr,
                               int64_t num_inner_items,
                               int64_t cur_offset) const;

        int64_t printOuterDim(int64_t dim,
                              int64_t flatten_dim,
                              void *print_ptr,
                              int64_t num_inner_items,
                              int64_t cur_offset) const;

        inline Printer(void *dev_ptr,
                       void *print_ptr,
                       TensorElementType type,
                       Span<const int64_t> dimensions,
                       int64_t num_total_bytes);

        void *dev_ptr_;
        void *print_ptr_;
        TensorElementType type_;
        int64_t num_dimensions_;
        std::array<int64_t, maxDimensions> dimensions_;
        int64_t num_total_bytes_;

    friend class Tensor;
    };

    Tensor(void *dev_ptr, TensorElementType type,
           Span<const int64_t> dimensions,
           Optional<int> gpu_id);

    Tensor(const Tensor &o);
    Tensor & operator=(const Tensor &o);
    
    inline void * devicePtr() const { return dev_ptr_; }
    inline TensorElementType type() const { return type_; }
    inline bool isOnGPU() const { return gpu_id_ != -1; }
    inline int gpuID() const { return gpu_id_; }
    inline int64_t numDims() const { return num_dimensions_; }
    inline const int64_t *dims() const { return dimensions_.data(); }
    int64_t numBytesPerItem() const;

    TensorInterface interface() const;

    Printer makePrinter() const;
private:
#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    void *dev_ptr_;
    TensorElementType type_;
    int gpu_id_;

    int64_t num_dimensions_;
    std::array<int64_t, maxDimensions> dimensions_;
};

struct NamedTensor {
    const char *name;
    Tensor tensor;
};

struct TrainStepInputInterface {
    Span<const NamedTensor> actions;
    Tensor resets;
    Tensor simCtrl;
    Span<const NamedTensor> pbt = {};
};

struct TrainStepOutputInterface {
    Span<const NamedTensor> observations;
    Tensor rewards;
    Tensor dones;
    Span<const NamedTensor> stats = {};
    Span<const NamedTensor> pbt = {};
};

struct TrainCheckpointingInterface {
    Tensor checkpointData;
};

class TrainInterface {
public:
    TrainInterface();
    TrainInterface(TrainStepInputInterface step_inputs,
                   TrainStepOutputInterface step_outputs,
                   Optional<TrainCheckpointingInterface> checkpointing = 
                       Optional<TrainCheckpointingInterface>::none());
    TrainInterface(TrainInterface &&o);
    ~TrainInterface();

    TrainInterface & operator=(TrainInterface &&o);

    TrainStepInputInterface stepInputs() const;
    TrainStepOutputInterface stepOutputs() const;
    Optional<TrainCheckpointingInterface> checkpointing() const;

    void cpuCopyStepInputs(void **buffers);
    void cpuCopyObservations(void **buffers);
    void cpuCopyStepOutputs(void **buffers);

#ifdef MADRONA_CUDA_SUPPORT
    void ** cudaCopyStepInputs(cudaStream_t strm, void **buffers);
    void cudaCopyObservations(cudaStream_t strm, void **buffers);
    void cudaCopyStepOutputs(cudaStream_t strm, void **buffers);
#endif

private:
    struct Impl;

#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    std::unique_ptr<Impl> impl_;
};


}
