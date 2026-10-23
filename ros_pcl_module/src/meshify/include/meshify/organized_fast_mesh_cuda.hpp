#pragma once

#include <vector>

namespace meshify
{

/// Unity clipping plane. valid == 0 means "no pose yet", same as the CPU node
/// skipping the filter. Normal is Unity local +Y in the pose frame.
struct UnityPlane
{
    int valid;
    double px;
    double py;
    double pz;
    double nx;
    double ny;
    double nz;
    double distance_threshold;
};

struct CudaMeshTimes
{
    long upload_ms;
    long mesh_ms;
    long compact_ms;
    long download_ms;
};

/// Create the CUDA context. Throws if no device is visible.
void initOrganizedFastMeshCuda();

/// Fast organized mesh, same call as setTrianglePixelSize(4) + TRIANGLE_RIGHT_CUT.
/// xyzrgb is width*height points, 6 floats each (x,y,z, r,g,b already /255).
/// triangles is the Float32MultiArray payload, 21 floats per triangle.
/// Throws on a CUDA error.
void runOrganizedFastMeshCuda(
    const float * xyzrgb,
    int width,
    int height,
    const UnityPlane & plane,
    std::vector<float> & triangles,
    CudaMeshTimes & times);

}  // namespace meshify
