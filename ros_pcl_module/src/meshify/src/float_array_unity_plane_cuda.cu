#include "meshify/organized_fast_mesh_cuda.hpp"

#include <cub/device/device_select.cuh>

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace meshify
{
namespace
{

// Float32ArrayPlaneNode::createOrganizedTriangulationMesh calls
// setTrianglePixelSize(4). PCL stores max(1, size - 1), so the walk is 3.
constexpr int kStride = 3;
// pcl::deg2rad(float) in PCL 1.12 and 1.14.
constexpr float kPclDeg2Rad = 0.017453293f;

struct Point6
{
    float x, y, z, r, g, b;
};

struct Triangle
{
    float f[21];
};

struct MeshArgs
{
    int width;
    int stride;
    float cos_angle_tolerance;
    int has_plane;
    double plane_px, plane_py, plane_pz;
    double plane_nx, plane_ny, plane_nz;
    double distance_threshold;
};

static_assert(sizeof(Point6) == 24, "point is 6 tightly packed floats");
static_assert(sizeof(Triangle) == 84, "wire triangle is 21 floats");

long nowMs()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void check(cudaError_t status, const char * what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ bool usable(const Point6 & p, const MeshArgs & args)
{
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z))
    {
        return false;
    }
    if (!args.has_plane)
    {
        return true;
    }
    // Same predicate as Float32ArrayPlaneNode::planeSegmentation:
    // signed distance = n · (p - plane_point), reject when <= threshold.
    const double dx = static_cast<double>(p.x) - args.plane_px;
    const double dy = static_cast<double>(p.y) - args.plane_py;
    const double dz = static_cast<double>(p.z) - args.plane_pz;
    const double sd = args.plane_nx * dx + args.plane_ny * dy + args.plane_nz * dz;
    return !(sd <= args.distance_threshold);
}

// OrganizedFastMesh::isShadowed with the defaults of the CPU node:
// viewpoint (0,0,0), 12.5 degrees, max edge length left unset.
__device__ bool edgeRejected(const Point6 & a, const Point6 & b, float cos_tol)
{
    const float ax = -a.x;
    const float ay = -a.y;
    const float az = -a.z;
    const float bx = b.x - a.x;
    const float by = b.y - a.y;
    const float bz = b.z - a.z;
    const float dist_a = sqrtf(ax * ax + ay * ay + az * az);
    const float dist_b = sqrtf(bx * bx + by * by + bz * bz);
    float cos_angle = (ax * bx + ay * by + az * bz) / (dist_a * dist_b);
    if (isnan(cos_angle))
    {
        cos_angle = 1.f;
    }
    return fabsf(cos_angle) >= cos_tol;
}

__device__ void writeVertex(float * d, const Point6 & p)
{
    d[0] = p.x;
    d[1] = p.y;
    d[2] = p.z;
    d[3] = p.r;
    d[4] = p.g;
    d[5] = p.b;
    d[6] = 1.f;
}

__device__ void emitOne(
    const Point6 * points, const MeshArgs & args,
    int ia, int ib, int ic, Triangle & slot, uint8_t & flag)
{
    const Point6 & a = points[ia];
    const Point6 & b = points[ib];
    const Point6 & c = points[ic];
    if (!usable(a, args) || !usable(b, args) || !usable(c, args) ||
        edgeRejected(a, b, args.cos_angle_tolerance) ||
        edgeRejected(b, c, args.cos_angle_tolerance) ||
        edgeRejected(c, a, args.cos_angle_tolerance))
    {
        flag = 0;
        return;
    }
    writeVertex(slot.f + 0, a);
    writeVertex(slot.f + 7, b);
    writeVertex(slot.f + 14, c);
    flag = 1;
}

// Right cut, same order as OrganizedFastMesh::makeRightCutMesh:
// (origin, down-right, right) then (origin, down, down-right).
__global__ void meshKernel(
    const Point6 * __restrict__ points, MeshArgs args,
    int cols, Triangle * __restrict__ slots, uint8_t * __restrict__ flags, int ncells)
{
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= ncells)
    {
        return;
    }
    const int row = cell / cols;
    const int col = cell - row * cols;
    const int x = col * args.stride;
    const int y = row * args.stride;
    const int i = y * args.width + x;
    const int i_right = i + args.stride;
    const int i_down = i + args.stride * args.width;
    const int i_down_right = i_down + args.stride;
    emitOne(points, args, i, i_down_right, i_right, slots[cell * 2], flags[cell * 2]);
    emitOne(points, args, i, i_down, i_down_right, slots[cell * 2 + 1], flags[cell * 2 + 1]);
}

struct Gpu
{
    Gpu()
    {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices <= 0)
        {
            throw std::runtime_error(
                "float_array_unity_plane_cuda_node: no CUDA device is visible");
        }
        check(cudaSetDevice(0), "cudaSetDevice");
        check(cudaFree(nullptr), "CUDA context init");
        const float rad = 12.5f * kPclDeg2Rad;
        cos_tol_ = std::fabs(std::cos(rad));
    }

    ~Gpu()
    {
        cudaFree(d_points_);
        cudaFree(d_slots_);
        cudaFree(d_flags_);
        cudaFree(d_selected_);
        cudaFree(d_count_);
        cudaFree(d_temp_);
    }

    Gpu(const Gpu &) = delete;
    Gpu & operator=(const Gpu &) = delete;

    void run(
        const float * xyzrgb, int width, int height, const UnityPlane & plane,
        std::vector<float> & triangles, CudaMeshTimes & times)
    {
        if (xyzrgb == nullptr || width < 2 || height < 2)
        {
            throw std::runtime_error("organized cloud needs width>=2 and height>=2");
        }

        // x < width - stride, same as makeRightCutMesh. A cloud no larger than the
        // stride has no cell.
        int cols = 0;
        int rows = 0;
        if (width > kStride && height > kStride)
        {
            cols = (width - kStride - 1) / kStride + 1;
            rows = (height - kStride - 1) / kStride + 1;
        }
        const int ncells = cols * rows;
        const int nslots = ncells * 2;
        const size_t npoints = static_cast<size_t>(width) * static_cast<size_t>(height);

        times = CudaMeshTimes{};
        long t = nowMs();
        ensurePoints(npoints);
        check(cudaMemcpy(
                  d_points_, xyzrgb, npoints * sizeof(Point6), cudaMemcpyHostToDevice),
              "upload point cloud");
        if (nslots > 0)
        {
            ensureSlots(nslots);
        }
        times.upload_ms = nowMs() - t;

        t = nowMs();
        if (ncells > 0)
        {
            MeshArgs args;
            args.width = width;
            args.stride = kStride;
            args.cos_angle_tolerance = cos_tol_;
            args.has_plane = plane.valid ? 1 : 0;
            args.plane_px = plane.px;
            args.plane_py = plane.py;
            args.plane_pz = plane.pz;
            args.plane_nx = plane.nx;
            args.plane_ny = plane.ny;
            args.plane_nz = plane.nz;
            args.distance_threshold = plane.distance_threshold;
            const int threads = 256;
            const int blocks = (ncells + threads - 1) / threads;
            meshKernel<<<blocks, threads>>>(d_points_, args, cols, d_slots_, d_flags_, ncells);
            check(cudaGetLastError(), "mesh kernel");
            check(cudaDeviceSynchronize(), "mesh kernel sync");
        }
        times.mesh_ms = nowMs() - t;

        t = nowMs();
        int count = 0;
        if (nslots > 0)
        {
            if (d_count_ == nullptr)
            {
                check(cudaMalloc(&d_count_, sizeof(int)), "cudaMalloc count");
            }
            size_t temp_bytes = 0;
            check(cub::DeviceSelect::Flagged(
                      nullptr, temp_bytes, d_slots_, d_flags_, d_selected_, d_count_, nslots),
                  "CUB size");
            if (temp_bytes > temp_bytes_)
            {
                cudaFree(d_temp_);
                d_temp_ = nullptr;
                temp_bytes_ = 0;
                check(cudaMalloc(&d_temp_, temp_bytes), "cudaMalloc CUB");
                temp_bytes_ = temp_bytes;
            }
            check(cub::DeviceSelect::Flagged(
                      d_temp_, temp_bytes_, d_slots_, d_flags_, d_selected_, d_count_, nslots),
                  "CUB DeviceSelect::Flagged");
            check(cudaDeviceSynchronize(), "compact sync");
            check(cudaMemcpy(&count, d_count_, sizeof(int), cudaMemcpyDeviceToHost), "download count");
            if (count < 0 || count > nslots)
            {
                throw std::runtime_error("CUDA compact returned an invalid triangle count");
            }
        }
        times.compact_ms = nowMs() - t;

        t = nowMs();
        triangles.resize(static_cast<size_t>(count) * 21);
        if (count > 0)
        {
            check(cudaMemcpy(
                      triangles.data(), d_selected_, static_cast<size_t>(count) * sizeof(Triangle),
                      cudaMemcpyDeviceToHost),
                  "download mesh");
        }
        times.download_ms = nowMs() - t;
    }

    void ensurePoints(size_t n)
    {
        if (d_points_ != nullptr && n <= points_cap_)
        {
            return;
        }
        cudaFree(d_points_);
        d_points_ = nullptr;
        points_cap_ = 0;
        check(cudaMalloc(&d_points_, n * sizeof(Point6)), "cudaMalloc points");
        points_cap_ = n;
    }

    void ensureSlots(int nslots)
    {
        if (d_slots_ != nullptr && nslots <= slots_cap_)
        {
            return;
        }
        cudaFree(d_slots_);
        cudaFree(d_flags_);
        cudaFree(d_selected_);
        cudaFree(d_temp_);
        d_slots_ = nullptr;
        d_flags_ = nullptr;
        d_selected_ = nullptr;
        d_temp_ = nullptr;
        slots_cap_ = 0;
        temp_bytes_ = 0;
        check(cudaMalloc(&d_slots_, static_cast<size_t>(nslots) * sizeof(Triangle)), "cudaMalloc slots");
        check(cudaMalloc(&d_flags_, static_cast<size_t>(nslots)), "cudaMalloc flags");
        check(cudaMalloc(&d_selected_, static_cast<size_t>(nslots) * sizeof(Triangle)), "cudaMalloc selected");
        slots_cap_ = nslots;
    }

    Point6 * d_points_ = nullptr;
    size_t points_cap_ = 0;
    Triangle * d_slots_ = nullptr;
    uint8_t * d_flags_ = nullptr;
    Triangle * d_selected_ = nullptr;
    int * d_count_ = nullptr;
    void * d_temp_ = nullptr;
    size_t temp_bytes_ = 0;
    int slots_cap_ = 0;
    float cos_tol_ = 1.f;
};

Gpu & gpu()
{
    static Gpu instance;
    return instance;
}

}  // namespace

void initOrganizedFastMeshCuda()
{
    gpu();
}

void runOrganizedFastMeshCuda(
    const float * xyzrgb, int width, int height, const UnityPlane & plane,
    std::vector<float> & triangles, CudaMeshTimes & times)
{
    gpu().run(xyzrgb, width, height, plane, triangles, times);
}

}  // namespace meshify
