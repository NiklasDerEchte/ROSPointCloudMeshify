#include "meshify/float_array_cuda.hpp"

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace meshify
{
namespace
{

// Float32ArrayNode::createOrganizedTriangulationMesh calls setTrianglePixelSize(4).
// PCL stores max(1, size - 1), so the walk is 3, TRIANGLE_RIGHT_CUT.
constexpr int kStride = 3;
constexpr float kPclDeg2Rad = 0.017453293f;
constexpr int kMaxIterations = 1000;
constexpr double kProbability = 0.85;
constexpr int kBatch = 32;
constexpr unsigned kSeed = 0x6A09E667u;
// Eigen::NumTraits<float>::dummy_precision(), used by SampleConsensusModelPlane.
constexpr float kCollinearEps = 1e-5f;

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
};

struct Moments
{
    double xx, xy, xz, yy, yz, zz, x, y, z;
    unsigned long long count;
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
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ unsigned xorshift(unsigned & s)
{
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

__device__ int warpSum(int v)
{
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// One block = one RANSAC hypothesis. Samples 3 finite points, fits a plane,
// counts inliers with |n·p + d| < threshold (same test as SACMODEL_PLANE).
__global__ void scoreKernel(
    const Point6 * __restrict__ points, const int * __restrict__ finite, int nfinite,
    int hyp0, int nhyp, double threshold, int * __restrict__ counts, float * __restrict__ coeffs)
{
    const int h = static_cast<int>(blockIdx.x);
    if (h >= nhyp) {
        return;
    }

    __shared__ float nx, ny, nz, d;
    __shared__ int valid;
    __shared__ int warp_sums[8];

    if (threadIdx.x == 0) {
        unsigned rng = kSeed ^ (static_cast<unsigned>(hyp0 + h) * 0x9E3779B9u);
        valid = 0;
        if (nfinite >= 3) {
            const int i0 = static_cast<int>(xorshift(rng) % static_cast<unsigned>(nfinite));
            int i1 = i0;
            int i2 = i0;
            for (int attempt = 0; attempt < 8 && i1 == i0; ++attempt) {
                i1 = static_cast<int>(xorshift(rng) % static_cast<unsigned>(nfinite));
            }
            for (int attempt = 0; attempt < 8 && (i2 == i0 || i2 == i1); ++attempt) {
                i2 = static_cast<int>(xorshift(rng) % static_cast<unsigned>(nfinite));
            }
            if (i0 != i1 && i1 != i2 && i0 != i2) {
                const Point6 a = points[finite[i0]];
                const Point6 b = points[finite[i1]];
                const Point6 c = points[finite[i2]];
                const float ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
                const float vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
                const float cx = uy * vz - uz * vy;
                const float cy = uz * vx - ux * vz;
                const float cz = ux * vy - uy * vx;
                const float cn = sqrtf(cx * cx + cy * cy + cz * cz);
                if (cn >= kCollinearEps) {
                    nx = cx / cn;
                    ny = cy / cn;
                    nz = cz / cn;
                    d = -(nx * a.x + ny * a.y + nz * a.z);
                    valid = 1;
                }
            }
        }
    }
    __syncthreads();

    int local = 0;
    if (valid) {
        for (int i = static_cast<int>(threadIdx.x); i < nfinite; i += static_cast<int>(blockDim.x)) {
            const Point6 p = points[finite[i]];
            const float dist = fabsf(nx * p.x + ny * p.y + nz * p.z + d);
            if (static_cast<double>(dist) < threshold) {
                ++local;
            }
        }
    }
    local = warpSum(local);
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x >> 5] = local;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        const int nwarps = static_cast<int>(blockDim.x) >> 5;
        for (int w = 0; w < nwarps; ++w) {
            total += warp_sums[w];
        }
        counts[h] = valid ? total : 0;
        coeffs[h * 4 + 0] = nx;
        coeffs[h * 4 + 1] = ny;
        coeffs[h * 4 + 2] = nz;
        coeffs[h * 4 + 3] = d;
    }
}

__global__ void finiteKernel(const Point6 * points, uint8_t * flags, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const Point6 p = points[i];
    flags[i] = (isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) ? 1 : 0;
}

__global__ void flagInliers(
    const Point6 * points, const float * coeff, double threshold,
    uint8_t * flags, int * first, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    const Point6 p = points[i];
    int in = 0;
    if (isfinite(p.x) && isfinite(p.y) && isfinite(p.z)) {
        const float dist = fabsf(coeff[0] * p.x + coeff[1] * p.y + coeff[2] * p.z + coeff[3]);
        if (static_cast<double>(dist) < threshold) {
            in = 1;
        }
    }
    flags[i] = static_cast<uint8_t>(in);
    if (in) {
        atomicMin(first, i);
    }
}

__global__ void loadShift(const Point6 * points, const int * first, float * k)
{
    const int i = *first;
    k[0] = points[i].x;
    k[1] = points[i].y;
    k[2] = points[i].z;
}

struct MomentOp
{
    const Point6 * points;
    const uint8_t * flags;
    const float * shift;

    __host__ __device__ Moments operator()(const int & i) const
    {
        Moments m{};
        if (!flags[i]) {
            return m;
        }
        const double x = static_cast<double>(points[i].x) - static_cast<double>(shift[0]);
        const double y = static_cast<double>(points[i].y) - static_cast<double>(shift[1]);
        const double z = static_cast<double>(points[i].z) - static_cast<double>(shift[2]);
        m.xx = x * x;
        m.xy = x * y;
        m.xz = x * z;
        m.yy = y * y;
        m.yz = y * z;
        m.zz = z * z;
        m.x = x;
        m.y = y;
        m.z = z;
        m.count = 1;
        return m;
    }
};

struct MomentSum
{
    __host__ __device__ Moments operator()(const Moments & a, const Moments & b) const
    {
        Moments c;
        c.xx = a.xx + b.xx;
        c.xy = a.xy + b.xy;
        c.xz = a.xz + b.xz;
        c.yy = a.yy + b.yy;
        c.yz = a.yz + b.yz;
        c.zz = a.zz + b.zz;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.count = a.count + b.count;
        return c;
    }
};

// Smallest eigenvector of the 3x3 covariance, same role as pcl::eigen33.
__device__ void jacobiSmallest(
    double a00, double a01, double a02, double a11, double a12, double a22, double n[3])
{
    double A[3][3] = {
        {a00, a01, a02},
        {a01, a11, a12},
        {a02, a12, a22}};
    double V[3][3] = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};
    for (int it = 0; it < 16; ++it) {
        int p = 0;
        int q = 1;
        double best = fabs(A[0][1]);
        if (fabs(A[0][2]) > best) {
            best = fabs(A[0][2]);
            p = 0;
            q = 2;
        }
        if (fabs(A[1][2]) > best) {
            best = fabs(A[1][2]);
            p = 1;
            q = 2;
        }
        if (best < 1e-15) {
            break;
        }
        const double app = A[p][p];
        const double aqq = A[q][q];
        const double apq = A[p][q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = copysign(1.0, tau) / (fabs(tau) + sqrt(1.0 + tau * tau));
        const double c = 1.0 / sqrt(1.0 + t * t);
        const double s = t * c;
        for (int r = 0; r < 3; ++r) {
            if (r == p || r == q) {
                continue;
            }
            const double arp = A[r][p];
            const double arq = A[r][q];
            A[r][p] = A[p][r] = c * arp - s * arq;
            A[r][q] = A[q][r] = s * arp + c * arq;
        }
        A[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p][q] = A[q][p] = 0.0;
        for (int r = 0; r < 3; ++r) {
            const double vrp = V[r][p];
            const double vrq = V[r][q];
            V[r][p] = c * vrp - s * vrq;
            V[r][q] = s * vrp + c * vrq;
        }
    }
    int k = 0;
    if (A[1][1] < A[k][k]) {
        k = 1;
    }
    if (A[2][2] < A[k][k]) {
        k = 2;
    }
    n[0] = V[0][k];
    n[1] = V[1][k];
    n[2] = V[2][k];
}

__global__ void eigenKernel(const Moments * moments, const float * shift, const float * fallback, float * out)
{
    const Moments m = *moments;
    if (m.count <= 3) {
        out[0] = fallback[0];
        out[1] = fallback[1];
        out[2] = fallback[2];
        out[3] = fallback[3];
        return;
    }
    const double inv = 1.0 / static_cast<double>(m.count);
    const double mx = m.x * inv;
    const double my = m.y * inv;
    const double mz = m.z * inv;
    double n[3];
    jacobiSmallest(
        m.xx * inv - mx * mx, m.xy * inv - mx * my, m.xz * inv - mx * mz,
        m.yy * inv - my * my, m.yz * inv - my * mz, m.zz * inv - mz * mz, n);
    const double norm = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (!(norm > 1e-12)) {
        out[0] = fallback[0];
        out[1] = fallback[1];
        out[2] = fallback[2];
        out[3] = fallback[3];
        return;
    }
    n[0] /= norm;
    n[1] /= norm;
    n[2] /= norm;
    const double cx = mx + static_cast<double>(shift[0]);
    const double cy = my + static_cast<double>(shift[1]);
    const double cz = mz + static_cast<double>(shift[2]);
    out[0] = static_cast<float>(n[0]);
    out[1] = static_cast<float>(n[1]);
    out[2] = static_cast<float>(n[2]);
    out[3] = static_cast<float>(-(n[0] * cx + n[1] * cy + n[2] * cz));
}

__global__ void applyInliers(Point6 * points, const float * coeff, double threshold, int n)
{
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return;
    }
    Point6 p = points[i];
    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        return;
    }
    const float dist = fabsf(coeff[0] * p.x + coeff[1] * p.y + coeff[2] * p.z + coeff[3]);
    if (static_cast<double>(dist) < threshold) {
        p.x = nanf("");
        p.y = nanf("");
        p.z = nanf("");
        points[i] = p;
    }
}

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
    if (isnan(cos_angle)) {
        cos_angle = 1.f;
    }
    return fabsf(cos_angle) >= cos_tol;
}

__device__ void writeVertex(float * dst, const Point6 & p)
{
    dst[0] = p.x;
    dst[1] = p.y;
    dst[2] = p.z;
    dst[3] = p.r;
    dst[4] = p.g;
    dst[5] = p.b;
    dst[6] = 1.f;
}

__device__ void emitOne(
    const Point6 * points, float cos_tol, int ia, int ib, int ic, Triangle & slot, uint8_t & flag)
{
    const Point6 & a = points[ia];
    const Point6 & b = points[ib];
    const Point6 & c = points[ic];
    if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z) ||
        !isfinite(b.x) || !isfinite(b.y) || !isfinite(b.z) ||
        !isfinite(c.x) || !isfinite(c.y) || !isfinite(c.z) ||
        edgeRejected(a, b, cos_tol) || edgeRejected(b, c, cos_tol) || edgeRejected(c, a, cos_tol))
    {
        flag = 0;
        return;
    }
    writeVertex(slot.f + 0, a);
    writeVertex(slot.f + 7, b);
    writeVertex(slot.f + 14, c);
    flag = 1;
}

__global__ void meshKernel(
    const Point6 * __restrict__ points, MeshArgs args,
    int cols, Triangle * __restrict__ slots, uint8_t * __restrict__ flags, int ncells)
{
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= ncells) {
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
    emitOne(points, args.cos_angle_tolerance, i, i_down_right, i_right, slots[cell * 2], flags[cell * 2]);
    emitOne(points, args.cos_angle_tolerance, i, i_down, i_down_right, slots[cell * 2 + 1], flags[cell * 2 + 1]);
}

int blocksFor(int n, int threads)
{
    return (n + threads - 1) / threads;
}

struct Gpu
{
    Gpu()
    {
        int devices = 0;
        const cudaError_t status = cudaGetDeviceCount(&devices);
        if (status != cudaSuccess || devices <= 0) {
            throw std::runtime_error("float_array_cuda_node: no CUDA device is visible");
        }
        check(cudaSetDevice(0), "cudaSetDevice");
        check(cudaFree(nullptr), "CUDA context init");
        const float rad = 12.5f * kPclDeg2Rad;
        cos_tol_ = std::fabs(std::cos(rad));
        check(cudaMalloc(&d_batch_counts_, static_cast<size_t>(kBatch) * sizeof(int)), "batch counts");
        check(cudaMalloc(&d_batch_coeffs_, static_cast<size_t>(kBatch) * 4 * sizeof(float)), "batch coeffs");
        check(cudaMalloc(&d_coeff_, 4 * sizeof(float)), "coeff");
        check(cudaMalloc(&d_opt_, 4 * sizeof(float)), "opt coeff");
        check(cudaMalloc(&d_shift_, 3 * sizeof(float)), "shift");
        check(cudaMalloc(&d_first_, sizeof(int)), "first");
        check(cudaMalloc(&d_nfinite_, sizeof(int)), "nfinite");
        check(cudaMalloc(&d_moments_, sizeof(Moments)), "moments");
        check(cudaMalloc(&d_tri_count_, sizeof(int)), "tri count");
    }

    ~Gpu()
    {
        cudaFree(d_points_);
        cudaFree(d_finite_flags_);
        cudaFree(d_finite_);
        cudaFree(d_batch_counts_);
        cudaFree(d_batch_coeffs_);
        cudaFree(d_coeff_);
        cudaFree(d_opt_);
        cudaFree(d_shift_);
        cudaFree(d_first_);
        cudaFree(d_nfinite_);
        cudaFree(d_moments_);
        cudaFree(d_temp_);
        cudaFree(d_slots_);
        cudaFree(d_tri_flags_);
        cudaFree(d_selected_);
        cudaFree(d_tri_count_);
    }

    Gpu(const Gpu &) = delete;
    Gpu & operator=(const Gpu &) = delete;

    CudaMeshStatus run(
        const float * xyzrgb, int width, int height, double threshold,
        std::vector<float> & triangles, CudaMeshTimes & times, std::size_t & nfinite_out)
    {
        if (xyzrgb == nullptr || width < 2 || height < 2) {
            throw std::runtime_error("organized cloud needs width>=2 and height>=2");
        }
        const std::size_t n64 = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        if (n64 > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("point cloud is larger than the CUDA index type");
        }
        const int n = static_cast<int>(n64);
        times = CudaMeshTimes{};
        nfinite_out = 0;

        const long t_ransac = nowMs();
        ensurePoints(n);
        check(cudaMemcpy(d_points_, xyzrgb, n64 * sizeof(Point6), cudaMemcpyHostToDevice), "upload");

        finiteKernel<<<blocksFor(n, 256), 256>>>(d_points_, d_finite_flags_, n);
        check(cudaGetLastError(), "finite kernel");

        cub::CountingInputIterator<int> ids(0);
        std::size_t select_bytes = 0;
        check(cub::DeviceSelect::Flagged(
                  nullptr, select_bytes, ids, d_finite_flags_, d_finite_, d_nfinite_, n),
              "CUB finite size");
        ensureTemp(select_bytes);
        check(cub::DeviceSelect::Flagged(
                  d_temp_, select_bytes, ids, d_finite_flags_, d_finite_, d_nfinite_, n),
              "CUB finite");
        int nfinite = 0;
        check(cudaMemcpy(&nfinite, d_nfinite_, sizeof(int), cudaMemcpyDeviceToHost), "nfinite");
        nfinite_out = static_cast<std::size_t>(std::max(nfinite, 0));
        if (nfinite < 3) {
            times.ransac_ms = nowMs() - t_ransac;
            triangles.clear();
            return CudaMeshStatus::NotEnoughFinite;
        }

        int best_count = 0;
        float best_c[4] = {0, 0, 1, 0};
        int iterations = 0;
        double k_limit = std::numeric_limits<double>::max();
        const double log_probability = std::log(1.0 - kProbability);
        std::vector<int> host_counts(static_cast<std::size_t>(kBatch));
        std::vector<float> host_coeffs(static_cast<std::size_t>(kBatch) * 4);
        while (iterations < kMaxIterations && static_cast<double>(iterations) <= k_limit) {
            const int batch = std::min(kBatch, kMaxIterations - iterations);
            scoreKernel<<<batch, 256>>>(
                d_points_, d_finite_, nfinite, iterations, batch, threshold,
                d_batch_counts_, d_batch_coeffs_);
            check(cudaGetLastError(), "score kernel");
            check(cudaMemcpy(
                      host_counts.data(), d_batch_counts_,
                      static_cast<std::size_t>(batch) * sizeof(int), cudaMemcpyDeviceToHost),
                  "score counts");
            check(cudaMemcpy(
                      host_coeffs.data(), d_batch_coeffs_,
                      static_cast<std::size_t>(batch) * 4 * sizeof(float), cudaMemcpyDeviceToHost),
                  "score coeffs");
            for (int h = 0; h < batch; ++h) {
                if (host_counts[static_cast<std::size_t>(h)] > best_count) {
                    best_count = host_counts[static_cast<std::size_t>(h)];
                    best_c[0] = host_coeffs[static_cast<std::size_t>(h) * 4 + 0];
                    best_c[1] = host_coeffs[static_cast<std::size_t>(h) * 4 + 1];
                    best_c[2] = host_coeffs[static_cast<std::size_t>(h) * 4 + 2];
                    best_c[3] = host_coeffs[static_cast<std::size_t>(h) * 4 + 3];
                }
            }
            if (best_count > 0) {
                const double w = static_cast<double>(best_count) / static_cast<double>(nfinite);
                double p_out = 1.0 - w * w * w;
                p_out = std::max(std::numeric_limits<double>::epsilon(), p_out);
                p_out = std::min(1.0 - std::numeric_limits<double>::epsilon(), p_out);
                k_limit = log_probability / std::log(p_out);
            }
            iterations += batch;
        }
        if (best_count <= 0) {
            times.ransac_ms = nowMs() - t_ransac;
            triangles.clear();
            return CudaMeshStatus::NoPlane;
        }

        check(cudaMemcpy(d_coeff_, best_c, 4 * sizeof(float), cudaMemcpyHostToDevice), "best coeff");
        const int first_init = std::numeric_limits<int>::max();
        check(cudaMemcpy(d_first_, &first_init, sizeof(int), cudaMemcpyHostToDevice), "reset first");
        flagInliers<<<blocksFor(n, 256), 256>>>(d_points_, d_coeff_, threshold, d_finite_flags_, d_first_, n);
        check(cudaGetLastError(), "flag inliers");
        loadShift<<<1, 1>>>(d_points_, d_first_, d_shift_);
        check(cudaGetLastError(), "load shift");

        MomentOp op{d_points_, d_finite_flags_, d_shift_};
        cub::CountingInputIterator<int> point_ids(0);
        cub::TransformInputIterator<Moments, MomentOp, cub::CountingInputIterator<int>> moment_in(point_ids, op);
        std::size_t reduce_bytes = 0;
        check(cub::DeviceReduce::Reduce(
                  nullptr, reduce_bytes, moment_in, d_moments_, n, MomentSum(), Moments{}),
              "CUB moments size");
        ensureTemp(reduce_bytes);
        check(cub::DeviceReduce::Reduce(
                  d_temp_, reduce_bytes, moment_in, d_moments_, n, MomentSum(), Moments{}),
              "CUB moments");
        eigenKernel<<<1, 1>>>(d_moments_, d_shift_, d_coeff_, d_opt_);
        check(cudaGetLastError(), "eigen");
        applyInliers<<<blocksFor(n, 256), 256>>>(d_points_, d_opt_, threshold, n);
        check(cudaGetLastError(), "apply inliers");
        check(cudaDeviceSynchronize(), "ransac sync");
        times.ransac_ms = nowMs() - t_ransac;

        const long t_mesh = nowMs();
        int cols = 0;
        int rows = 0;
        if (width > kStride && height > kStride) {
            cols = (width - kStride - 1) / kStride + 1;
            rows = (height - kStride - 1) / kStride + 1;
        }
        const int ncells = cols * rows;
        const int nslots = ncells * 2;
        int tri_count = 0;
        if (ncells > 0) {
            ensureSlots(nslots);
            MeshArgs args;
            args.width = width;
            args.stride = kStride;
            args.cos_angle_tolerance = cos_tol_;
            meshKernel<<<blocksFor(ncells, 256), 256>>>(
                d_points_, args, cols, d_slots_, d_tri_flags_, ncells);
            check(cudaGetLastError(), "mesh kernel");
            std::size_t tri_bytes = 0;
            check(cub::DeviceSelect::Flagged(
                      nullptr, tri_bytes, d_slots_, d_tri_flags_, d_selected_, d_tri_count_, nslots),
                  "CUB triangles size");
            ensureTemp(tri_bytes);
            check(cub::DeviceSelect::Flagged(
                      d_temp_, tri_bytes, d_slots_, d_tri_flags_, d_selected_, d_tri_count_, nslots),
                  "CUB triangles");
            check(cudaMemcpy(&tri_count, d_tri_count_, sizeof(int), cudaMemcpyDeviceToHost), "tri count");
            if (tri_count < 0 || tri_count > nslots) {
                throw std::runtime_error("CUDA compact returned an invalid triangle count");
            }
        }
        triangles.resize(static_cast<std::size_t>(tri_count) * 21);
        if (tri_count > 0) {
            check(cudaMemcpy(
                      triangles.data(), d_selected_,
                      static_cast<std::size_t>(tri_count) * sizeof(Triangle), cudaMemcpyDeviceToHost),
                  "download mesh");
        }
        times.mesh_ms = nowMs() - t_mesh;
        if (tri_count == 0) {
            return CudaMeshStatus::EmptyMesh;
        }
        return CudaMeshStatus::Ok;
    }

    void ensureTemp(std::size_t bytes)
    {
        if (bytes <= temp_cap_) {
            return;
        }
        cudaFree(d_temp_);
        d_temp_ = nullptr;
        temp_cap_ = 0;
        check(cudaMalloc(&d_temp_, bytes), "cudaMalloc temp");
        temp_cap_ = bytes;
    }

    void ensurePoints(int n)
    {
        if (d_points_ != nullptr && n <= points_cap_) {
            return;
        }
        cudaFree(d_points_);
        cudaFree(d_finite_flags_);
        cudaFree(d_finite_);
        d_points_ = nullptr;
        d_finite_flags_ = nullptr;
        d_finite_ = nullptr;
        points_cap_ = 0;
        check(cudaMalloc(&d_points_, static_cast<std::size_t>(n) * sizeof(Point6)), "points");
        check(cudaMalloc(&d_finite_flags_, static_cast<std::size_t>(n)), "finite flags");
        check(cudaMalloc(&d_finite_, static_cast<std::size_t>(n) * sizeof(int)), "finite index");
        points_cap_ = n;
    }

    void ensureSlots(int nslots)
    {
        if (d_slots_ != nullptr && nslots <= slots_cap_) {
            return;
        }
        cudaFree(d_slots_);
        cudaFree(d_tri_flags_);
        cudaFree(d_selected_);
        d_slots_ = nullptr;
        d_tri_flags_ = nullptr;
        d_selected_ = nullptr;
        slots_cap_ = 0;
        check(cudaMalloc(&d_slots_, static_cast<std::size_t>(nslots) * sizeof(Triangle)), "slots");
        check(cudaMalloc(&d_tri_flags_, static_cast<std::size_t>(nslots)), "tri flags");
        check(cudaMalloc(&d_selected_, static_cast<std::size_t>(nslots) * sizeof(Triangle)), "selected");
        slots_cap_ = nslots;
    }

    Point6 * d_points_ = nullptr;
    uint8_t * d_finite_flags_ = nullptr;
    int * d_finite_ = nullptr;
    int points_cap_ = 0;
    int * d_batch_counts_ = nullptr;
    float * d_batch_coeffs_ = nullptr;
    float * d_coeff_ = nullptr;
    float * d_opt_ = nullptr;
    float * d_shift_ = nullptr;
    int * d_first_ = nullptr;
    int * d_nfinite_ = nullptr;
    Moments * d_moments_ = nullptr;
    void * d_temp_ = nullptr;
    std::size_t temp_cap_ = 0;
    Triangle * d_slots_ = nullptr;
    uint8_t * d_tri_flags_ = nullptr;
    Triangle * d_selected_ = nullptr;
    int * d_tri_count_ = nullptr;
    int slots_cap_ = 0;
    float cos_tol_ = 1.f;
};

Gpu & gpu()
{
    static Gpu instance;
    return instance;
}

}  // namespace

void initFloatArrayCuda()
{
    gpu();
}

CudaMeshStatus runFloatArrayCuda(
    const float * xyzrgb, int width, int height, double distance_threshold,
    std::vector<float> & triangles, CudaMeshTimes & times, std::size_t & nfinite)
{
    return gpu().run(xyzrgb, width, height, distance_threshold, triangles, times, nfinite);
}

}  // namespace meshify
