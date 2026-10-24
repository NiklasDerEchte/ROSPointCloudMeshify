#include "meshify/float_array_cuda.hpp"

#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/organized_fast_mesh.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace
{

constexpr float kPlaneZ = 0.55f;
constexpr float kObjectZ = 0.80f;
constexpr double kThreshold = 0.01;

pcl::PointCloud<pcl::PointXYZRGB> makeCloud()
{
    const int width = 72;
    const int height = 48;
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.width = static_cast<uint32_t>(width);
    cloud.height = static_cast<uint32_t>(height);
    cloud.is_dense = false;
    cloud.points.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            auto & p = cloud.at(x, y);
            p.x = (x - width / 2) * 0.004f;
            p.y = (y - height / 2) * 0.004f;
            const bool object = x >= 24 && x < 48 && y >= 16 && y < 36;
            p.z = object ? kObjectZ : kPlaneZ;
            if (x == 3 && y == 3) {
                p.x = std::numeric_limits<float>::quiet_NaN();
                p.y = std::numeric_limits<float>::quiet_NaN();
                p.z = std::numeric_limits<float>::quiet_NaN();
            }
            if (x == 9 && y == 9) {
                p.z = std::numeric_limits<float>::infinity();
            }
            p.r = static_cast<uint8_t>(x * 3);
            p.g = static_cast<uint8_t>(y * 5);
            p.b = 40;
            p.a = 255;
        }
    }
    return cloud;
}

std::vector<float> pack(const pcl::PointCloud<pcl::PointXYZRGB> & cloud)
{
    std::vector<float> out(cloud.points.size() * 6);
    for (size_t i = 0; i < cloud.points.size(); ++i) {
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

std::vector<float> cpuMesh(pcl::PointCloud<pcl::PointXYZRGB> cloud)
{
    pcl::IndicesPtr finite(new pcl::Indices());
    finite->reserve(cloud.size());
    for (std::size_t i = 0; i < cloud.points.size(); ++i) {
        const auto & p = cloud.points[i];
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            finite->push_back(static_cast<int>(i));
        }
    }
    auto input = cloud.makeShared();
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    seg.setOptimizeCoefficients(true);
    seg.setMaxIterations(1000);
    seg.setProbability(0.85);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(kThreshold);
    seg.setInputCloud(input);
    seg.setIndices(finite);
    seg.segment(*inliers, *coefficients);
    EXPECT_FALSE(inliers->indices.empty());
    for (int idx : inliers->indices) {
        cloud.points[static_cast<size_t>(idx)].x = std::numeric_limits<float>::quiet_NaN();
        cloud.points[static_cast<size_t>(idx)].y = std::numeric_limits<float>::quiet_NaN();
        cloud.points[static_cast<size_t>(idx)].z = std::numeric_limits<float>::quiet_NaN();
    }

    auto meshed = cloud.makeShared();
    pcl::OrganizedFastMesh<pcl::PointXYZRGB> ofm;
    ofm.setInputCloud(meshed);
    ofm.setTrianglePixelSize(4);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZRGB>::TRIANGLE_RIGHT_CUT);
    pcl::PolygonMesh mesh;
    ofm.reconstruct(mesh);

    std::vector<float> out;
    out.reserve(mesh.polygons.size() * 21);
    for (const auto & poly : mesh.polygons) {
        if (poly.vertices.size() != 3) {
            continue;
        }
        for (const auto index : poly.vertices) {
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

void expectPlaneGone(const std::vector<float> & mesh)
{
    ASSERT_GT(mesh.size(), 0u);
    ASSERT_EQ(mesh.size() % 21, 0u);
    for (size_t i = 0; i < mesh.size(); i += 7) {
        EXPECT_GT(std::fabs(mesh[i + 2] - kPlaneZ), static_cast<float>(kThreshold))
            << "vertex " << (i / 7) << " still lies on the plane";
    }
}

}  // namespace

TEST(FloatArrayCuda, RemovesPlaneAndMatchesOrganizedFastMesh)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
        GTEST_SKIP() << "no CUDA device, GPU mesh was not executed";
    }

    const auto cloud = makeCloud();
    const std::vector<float> cpu = cpuMesh(cloud);
    ASSERT_GT(cpu.size(), 0u);
    expectPlaneGone(cpu);

    const std::vector<float> xyzrgb = pack(cloud);
    std::vector<float> gpu;
    meshify::CudaMeshTimes times{};
    std::size_t nfinite = 0;
    const meshify::CudaMeshStatus status = meshify::runFloatArrayCuda(
        xyzrgb.data(), static_cast<int>(cloud.width), static_cast<int>(cloud.height),
        kThreshold, gpu, times, nfinite);
    ASSERT_EQ(status, meshify::CudaMeshStatus::Ok);
    EXPECT_GT(nfinite, 3u);
    expectPlaneGone(gpu);

    if (gpu.size() == cpu.size()) {
        for (size_t i = 0; i < gpu.size(); ++i) {
            ASSERT_EQ(gpu[i], cpu[i]) << "float " << i << " triangle " << (i / 21);
        }
    } else {
        const double rel = std::fabs(static_cast<double>(gpu.size()) - static_cast<double>(cpu.size())) /
                           static_cast<double>(cpu.size());
        EXPECT_LT(rel, 0.02) << "gpu triangles " << (gpu.size() / 21)
                             << " cpu triangles " << (cpu.size() / 21);
    }
}

TEST(FloatArrayCuda, NotEnoughFinitePoints)
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
        GTEST_SKIP() << "no CUDA device, GPU mesh was not executed";
    }

    const int width = 4;
    const int height = 4;
    std::vector<float> xyzrgb(static_cast<size_t>(width * height) * 6, std::numeric_limits<float>::quiet_NaN());
    xyzrgb[0] = 0.f;
    xyzrgb[1] = 0.f;
    xyzrgb[2] = 0.5f;
    xyzrgb[6] = 0.01f;
    xyzrgb[7] = 0.f;
    xyzrgb[8] = 0.5f;

    std::vector<float> triangles;
    meshify::CudaMeshTimes times{};
    std::size_t nfinite = 99;
    const meshify::CudaMeshStatus status = meshify::runFloatArrayCuda(
        xyzrgb.data(), width, height, kThreshold, triangles, times, nfinite);
    EXPECT_EQ(status, meshify::CudaMeshStatus::NotEnoughFinite);
    EXPECT_EQ(nfinite, 2u);
    EXPECT_TRUE(triangles.empty());
}
