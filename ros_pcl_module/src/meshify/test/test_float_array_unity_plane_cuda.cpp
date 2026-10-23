#include "meshify/organized_fast_mesh_cuda.hpp"

#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/organized_fast_mesh.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

pcl::PointCloud<pcl::PointXYZRGB> makeCloud(int width, int height)
{
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = static_cast<uint32_t>(width);
    cloud.height = static_cast<uint32_t>(height);
    cloud.is_dense = false;
    cloud.points.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            auto & p = cloud.at(x, y);
            p.x = (x - width / 2) * 0.004f;
            p.y = (y - height / 2) * 0.004f;
            // Step across x so the stride-3 walk still sees a long edge.
            p.z = (x >= width / 2) ? 0.20f : 0.85f;
            if (x == 6 && y == 6)
            {
                p.x = std::numeric_limits<float>::quiet_NaN();
                p.y = std::numeric_limits<float>::quiet_NaN();
                p.z = std::numeric_limits<float>::quiet_NaN();
            }
            if (x == 3 && y == 3)
            {
                p.z = std::numeric_limits<float>::infinity();
            }
            p.r = static_cast<uint8_t>((x * 17) & 255);
            p.g = static_cast<uint8_t>((y * 29) & 255);
            p.b = static_cast<uint8_t>((x + 3 * y) & 255);
            p.a = 255;
        }
    }
    return cloud;
}

std::vector<float> pack(const pcl::PointCloud<pcl::PointXYZRGB> & cloud)
{
    std::vector<float> out(cloud.points.size() * 6);
    for (size_t i = 0; i < cloud.points.size(); ++i)
    {
        const auto & p = cloud.points[i];
        out[i * 6 + 0] = p.x;
        out[i * 6 + 1] = p.y;
        out[i * 6 + 2] = p.z;
        out[i * 6 + 3] = static_cast<float>(p.r) / 255.f;
        out[i * 6 + 4] = static_cast<float>(p.g) / 255.f;
        out[i * 6 + 5] = static_cast<float>(p.b) / 255.f;
    }
    return out;
}

std::vector<float> runOfm(pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
    auto input = cloud.makeShared();
    pcl::OrganizedFastMesh<pcl::PointXYZRGB> ofm;
    ofm.setInputCloud(input);
    ofm.setTrianglePixelSize(4);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZRGB>::TRIANGLE_RIGHT_CUT);
    pcl::PolygonMesh mesh;
    ofm.reconstruct(mesh);

    std::vector<float> out;
    out.reserve(mesh.polygons.size() * 21);
    for (const auto & poly : mesh.polygons)
    {
        if (poly.vertices.size() != 3)
        {
            continue;
        }
        for (const auto index : poly.vertices)
        {
            const auto & p = cloud.points[index];
            out.push_back(p.x);
            out.push_back(p.y);
            out.push_back(p.z);
            out.push_back(static_cast<float>(p.r) / 255.f);
            out.push_back(static_cast<float>(p.g) / 255.f);
            out.push_back(static_cast<float>(p.b) / 255.f);
            out.push_back(1.f);
        }
    }
    return out;
}

void expectSame(const std::vector<float> & gpu, const std::vector<float> & pcl_mesh)
{
    ASSERT_EQ(gpu.size() % 21, 0u);
    ASSERT_EQ(gpu.size(), pcl_mesh.size())
        << (gpu.size() / 21) << " gpu triangles vs " << (pcl_mesh.size() / 21) << " pcl triangles";
    ASSERT_GT(gpu.size(), 0u);
    for (size_t i = 0; i < gpu.size(); ++i)
    {
        ASSERT_EQ(gpu[i], pcl_mesh[i]) << "float " << i << " triangle " << (i / 21);
    }
}

}  // namespace

TEST(Float32ArrayPlaneCuda, MatchesOrganizedFastMesh)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0)
    {
        GTEST_SKIP() << "no CUDA device, GPU mesh was not executed";
    }

    const auto cloud = makeCloud(48, 36);
    const std::vector<float> xyzrgb = pack(cloud);

    meshify::UnityPlane none{};
    meshify::CudaMeshTimes times;
    std::vector<float> gpu;
    meshify::runOrganizedFastMeshCuda(
        xyzrgb.data(), 48, 36, none, gpu, times);
    expectSame(gpu, runOfm(cloud));

    // Same cut as planeSegmentation: n · (p - point) <= threshold becomes NaN,
    // then OrganizedFastMesh runs on what is left.
    meshify::UnityPlane plane{};
    plane.valid = 1;
    plane.ny = 1.0;
    plane.distance_threshold = 0.0;
    auto clipped = cloud;
    for (auto & p : clipped.points)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        {
            continue;
        }
        const double sd = plane.ny * (static_cast<double>(p.y) - plane.py);
        if (sd <= plane.distance_threshold)
        {
            p.x = std::numeric_limits<float>::quiet_NaN();
            p.y = std::numeric_limits<float>::quiet_NaN();
            p.z = std::numeric_limits<float>::quiet_NaN();
        }
    }
    std::vector<float> gpu_plane;
    meshify::runOrganizedFastMeshCuda(
        xyzrgb.data(), 48, 36, plane, gpu_plane, times);
    expectSame(gpu_plane, runOfm(clipped));
    EXPECT_LT(gpu_plane.size(), gpu.size());
}
