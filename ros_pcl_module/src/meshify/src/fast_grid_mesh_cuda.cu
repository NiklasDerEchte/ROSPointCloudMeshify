#include "meshify/fast_grid_mesh.hpp"

#include "meshify/grid_eval.hpp"

#include <cub/device/device_select.cuh>

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace meshify
{
namespace
{

using Clock = std::chrono::steady_clock;

int64_t us_between(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

void cuda_check(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

__global__ void grid_mesh_kernel(
  const PackedPoint * __restrict__ points,
  MeshConfig cfg,
  int cols,
  TriangleRec * __restrict__ slots,
  uint8_t * __restrict__ flags,
  int ncells)
{
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell >= ncells) {
    return;
  }
  const int row = cell / cols;
  const int col = cell - row * cols;
  emit_cell(
    points, cfg, col * cfg.stride, row * cfg.stride,
    slots + static_cast<size_t>(cell) * 2,
    flags + static_cast<size_t>(cell) * 2);
}

class CudaGridMesher : public GridMesher
{
public:
  CudaGridMesher()
  {
    int devices = 0;
    cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices <= 0) {
      throw std::runtime_error("CUDA backend: no CUDA device is visible");
    }
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    cuda_check(cudaFree(nullptr), "CUDA context init");
    cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
  }

  CudaGridMesher(const CudaGridMesher &) = delete;
  CudaGridMesher & operator=(const CudaGridMesher &) = delete;

  ~CudaGridMesher() override
  {
    cudaFree(d_points_);
    cudaFree(d_slots_);
    cudaFree(d_flags_);
    cudaFree(d_selected_);
    cudaFree(d_count_);
    cudaFree(d_temp_);
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  const char * name() const override {return "cuda";}

  StageMicros mesh(
    const PackedCloud & cloud, const MeshConfig & cfg, std::vector<float> & xyzrgba) override
  {
    if (cfg.width != cloud.width || cfg.height != cloud.height) {
      throw std::runtime_error("mesh config dimensions do not match the packed cloud");
    }
    const size_t expect =
      static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
    if (cloud.points.size() != expect) {
      throw std::runtime_error("packed cloud size does not match width*height");
    }

    const GridShape grid = make_grid_shape(cloud.width, cloud.height, cfg.stride);
    const int nslots = grid.ncells * 2;
    StageMicros times;

    const auto t0 = Clock::now();
    if (!cloud.points.empty()) {
      ensure_points(cloud.points.size());
      cuda_check(
        cudaMemcpyAsync(
          d_points_, cloud.points.data(), cloud.points.size() * sizeof(PackedPoint),
          cudaMemcpyHostToDevice, stream_),
        "HtoD points");
    }
    if (nslots > 0) {
      ensure_slots(nslots);
    }
    cuda_check(cudaStreamSynchronize(stream_), "sync upload");
    const auto t1 = Clock::now();
    times.upload_us = us_between(t0, t1);

    if (nslots > 0) {
      constexpr int threads = 256;
      const int blocks = (grid.ncells + threads - 1) / threads;
      grid_mesh_kernel<<<blocks, threads, 0, stream_>>>(
        d_points_, cfg, grid.cols, d_slots_, d_flags_, grid.ncells);
      cuda_check(cudaGetLastError(), "mesh kernel launch");
      cuda_check(cudaStreamSynchronize(stream_), "sync mesh");
    }
    const auto t2 = Clock::now();
    times.clip_mesh_us = us_between(t1, t2);

    if (nslots > 0) {
      if (d_count_ == nullptr) {
        cuda_check(cudaMalloc(&d_count_, sizeof(int)), "cudaMalloc count");
      }
      size_t temp_bytes = 0;
      cuda_check(
        cub::DeviceSelect::Flagged(
          nullptr, temp_bytes, d_slots_, d_flags_, d_selected_, d_count_, nslots, stream_),
        "CUB DeviceSelect::Flagged size");
      if (temp_bytes > temp_bytes_) {
        cudaFree(d_temp_);
        d_temp_ = nullptr;
        temp_bytes_ = 0;
        cuda_check(cudaMalloc(&d_temp_, temp_bytes), "cudaMalloc CUB temp");
        temp_bytes_ = temp_bytes;
      }
      cuda_check(
        cub::DeviceSelect::Flagged(
          d_temp_, temp_bytes, d_slots_, d_flags_, d_selected_, d_count_, nslots, stream_),
        "CUB DeviceSelect::Flagged");
      cuda_check(cudaStreamSynchronize(stream_), "sync compact");
    }
    const auto t3 = Clock::now();
    times.compact_us = us_between(t2, t3);

    int count = 0;
    if (nslots > 0) {
      cuda_check(
        cudaMemcpyAsync(&count, d_count_, sizeof(int), cudaMemcpyDeviceToHost, stream_),
        "DtoH count");
      cuda_check(cudaStreamSynchronize(stream_), "sync count");
      if (count < 0 || count > nslots) {
        throw std::runtime_error("CUDA compact returned an invalid triangle count");
      }
      xyzrgba.resize(static_cast<size_t>(count) * 21);
      if (count > 0) {
        cuda_check(
          cudaMemcpyAsync(
            xyzrgba.data(), d_selected_, static_cast<size_t>(count) * sizeof(TriangleRec),
            cudaMemcpyDeviceToHost, stream_),
          "DtoH triangles");
        cuda_check(cudaStreamSynchronize(stream_), "sync download");
      }
    } else {
      xyzrgba.clear();
    }
    const auto t4 = Clock::now();
    times.download_us = us_between(t3, t4);
    return times;
  }

private:
  void ensure_points(size_t n)
  {
    if (d_points_ != nullptr && n <= points_cap_) {
      return;
    }
    cudaFree(d_points_);
    d_points_ = nullptr;
    points_cap_ = 0;
    cuda_check(cudaMalloc(&d_points_, n * sizeof(PackedPoint)), "cudaMalloc points");
    points_cap_ = n;
  }

  void ensure_slots(int nslots)
  {
    if (d_slots_ != nullptr && nslots <= slots_cap_) {
      return;
    }
    cudaFree(d_slots_);
    cudaFree(d_flags_);
    cudaFree(d_selected_);
    d_slots_ = nullptr;
    d_flags_ = nullptr;
    d_selected_ = nullptr;
    slots_cap_ = 0;
    cudaFree(d_temp_);
    d_temp_ = nullptr;
    temp_bytes_ = 0;
    cuda_check(cudaMalloc(&d_slots_, static_cast<size_t>(nslots) * sizeof(TriangleRec)), "cudaMalloc slots");
    cuda_check(cudaMalloc(&d_flags_, static_cast<size_t>(nslots) * sizeof(uint8_t)), "cudaMalloc flags");
    cuda_check(
      cudaMalloc(&d_selected_, static_cast<size_t>(nslots) * sizeof(TriangleRec)),
      "cudaMalloc selected");
    slots_cap_ = nslots;
  }

  cudaStream_t stream_ = nullptr;
  PackedPoint * d_points_ = nullptr;
  size_t points_cap_ = 0;
  TriangleRec * d_slots_ = nullptr;
  uint8_t * d_flags_ = nullptr;
  TriangleRec * d_selected_ = nullptr;
  int * d_count_ = nullptr;
  void * d_temp_ = nullptr;
  size_t temp_bytes_ = 0;
  int slots_cap_ = 0;
};

}  // namespace

std::unique_ptr<GridMesher> create_cuda_grid_mesher()
{
  return std::make_unique<CudaGridMesher>();
}

int available_cuda_devices()
{
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 0) {
    return 0;
  }
  return n;
}

}  // namespace meshify
