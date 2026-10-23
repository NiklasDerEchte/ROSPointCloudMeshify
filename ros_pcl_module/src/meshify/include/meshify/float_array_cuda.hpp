#pragma once

#include <cstddef>
#include <vector>

namespace meshify
{

struct CudaMeshTimes
{
    long ransac_ms;
    long mesh_ms;
};

enum class CudaMeshStatus
{
    Ok,
    NotEnoughFinite,
    NoPlane,
    EmptyMesh
};

/// Create the CUDA context. Throws if no device is visible.
void initFloatArrayCuda();

/// Fast path of Float32ArrayNode: RANSAC plane (finite points only, threshold,
/// 1000 iterations, probability 0.85, optimized coefficients), inliers set to
/// NaN, then OrganizedFastMesh right-cut. xyzrgb is width*height points, 6
/// floats each (x,y,z and r,g,b already /255). triangles is the
/// Float32MultiArray payload, 21 floats per triangle.
/// Throws on a CUDA error.
CudaMeshStatus runFloatArrayCuda(
    const float * xyzrgb,
    int width,
    int height,
    double distance_threshold,
    std::vector<float> & triangles,
    CudaMeshTimes & times,
    std::size_t & nfinite);

}  // namespace meshify
