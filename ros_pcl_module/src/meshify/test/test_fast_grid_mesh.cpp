#include "meshify/fast_grid_mesh.hpp"
#include "meshify/grid_eval.hpp"

#include <pcl/common/angles.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_field.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

using meshify::MeshConfig;
using meshify::MeshRequest;
using meshify::PackedCloud;
using meshify::PackedPoint;

bool same_float(float a, float b)
{
  if (std::isnan(a) && std::isnan(b)) {
    return true;
  }
  return a == b;
}

testing::AssertionResult same_mesh(const std::vector<float> & a, const std::vector<float> & b)
{
  if (a.size() != b.size() || a.size() % 21 != 0) {
    return testing::AssertionFailure()
           << "float counts " << a.size() << " vs " << b.size()
           << " (" << (a.size() / 21) << " vs " << (b.size() / 21) << " triangles)";
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return testing::AssertionFailure()
             << "mismatch at float " << i << " triangle " << (i / 21)
             << " component " << (i % 21) << " a=" << a[i] << " b=" << b[i];
    }
  }
  return testing::AssertionSuccess();
}

int brute_cells(int width, int height, int stride)
{
  if (stride < 1 || width <= stride || height <= stride) {
    return 0;
  }
  int n = 0;
  for (int y = 0; y < height - stride; y += stride) {
    for (int x = 0; x < width - stride; x += stride) {
      ++n;
    }
  }
  return n;
}

pcl::PointCloud<pcl::PointXYZRGB> make_cloud(int width, int height)
{
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
      // Half-plane step so every stride that crosses width/2 sees a long edge.
      // A single column at width/2 falls between the samples of stride 3.
      p.z = (x >= width / 2) ? 0.20f : 0.85f;
      if ((x == 6 && y == 6) || (x == 2 && y == 2) || (x == 8 && y == 4)) {
        p.x = std::numeric_limits<float>::quiet_NaN();
        p.y = std::numeric_limits<float>::quiet_NaN();
        p.z = std::numeric_limits<float>::quiet_NaN();
      }
      if (x == 3 && y == 3) {
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

PackedCloud pack_from_pcl(const pcl::PointCloud<pcl::PointXYZRGB> & cloud)
{
  PackedCloud packed;
  packed.width = static_cast<int>(cloud.width);
  packed.height = static_cast<int>(cloud.height);
  packed.has_color = true;
  packed.points.resize(cloud.points.size());
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    const auto & p = cloud.points[i];
    PackedPoint & dst = packed.points[i];
    dst.x = p.x;
    dst.y = p.y;
    dst.z = p.z;
    dst.r = static_cast<float>(p.r) / 255.f;
    dst.g = static_cast<float>(p.g) / 255.f;
    dst.b = static_cast<float>(p.b) / 255.f;
  }
  return packed;
}

std::vector<float> expand_mesh(
  const pcl::PolygonMesh & mesh, const pcl::PointCloud<pcl::PointXYZRGB> & cloud)
{
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

std::vector<float> run_ofm(
  const pcl::PointCloud<pcl::PointXYZRGB> & source, const MeshRequest & req, bool legacy_call)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud = source;
  if (req.has_plane) {
    const MeshConfig cfg = meshify::make_mesh_config(
      static_cast<int>(cloud.width), static_cast<int>(cloud.height), req);
    for (auto & p : cloud.points) {
      PackedPoint pp;
      pp.x = p.x;
      pp.y = p.y;
      pp.z = p.z;
      if (!meshify::point_usable(pp, cfg)) {
        p.x = std::numeric_limits<float>::quiet_NaN();
        p.y = std::numeric_limits<float>::quiet_NaN();
        p.z = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }

  auto input = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>(std::move(cloud));
  pcl::OrganizedFastMesh<pcl::PointXYZRGB> ofm;
  ofm.setInputCloud(input);
  if (legacy_call) {
    ofm.setTrianglePixelSize(4);
  } else if (req.grid_stride > 0) {
    ofm.setTrianglePixelSize(req.grid_stride + 1);
  } else {
    ofm.setTrianglePixelSize(std::max(1, req.triangle_pixel_size));
  }
  ofm.setTriangulationType(pcl::OrganizedFastMesh<pcl::PointXYZRGB>::TRIANGLE_RIGHT_CUT);
  if (!legacy_call) {
    ofm.setViewpoint(Eigen::Vector3f(req.viewpoint_x, req.viewpoint_y, req.viewpoint_z));
    if (req.angle_tolerance_deg > 0.f) {
      ofm.setAngleTolerance(pcl::deg2rad(req.angle_tolerance_deg));
    } else {
      ofm.setAngleTolerance(-1.f);
    }
    ofm.setMaxEdgeLength(req.max_edge_a, req.max_edge_b, req.max_edge_c);
    ofm.storeShadowedFaces(req.store_shadowed_faces);
    ofm.useDepthAsDistance(req.use_depth_as_distance);
  }
  pcl::PolygonMesh mesh;
  ofm.reconstruct(mesh);
  return expand_mesh(mesh, *input);
}

std::vector<float> run_backend(
  meshify::GridMesher & mesher, const pcl::PointCloud<pcl::PointXYZRGB> & cloud, const MeshRequest & req)
{
  const PackedCloud packed = pack_from_pcl(cloud);
  const MeshConfig cfg = meshify::make_mesh_config(packed.width, packed.height, req);
  std::vector<float> out;
  mesher.mesh(packed, cfg, out);
  return out;
}

void expect_matches_pcl(
  meshify::GridMesher & cpu, meshify::GridMesher & omp,
  const pcl::PointCloud<pcl::PointXYZRGB> & cloud, const MeshRequest & req, bool legacy_call)
{
  const std::vector<float> reference = run_ofm(cloud, req, legacy_call);
  const std::vector<float> cpu_out = run_backend(cpu, cloud, req);
  const std::vector<float> omp_out = run_backend(omp, cloud, req);
  ASSERT_TRUE(same_mesh(reference, cpu_out));
  ASSERT_TRUE(same_mesh(cpu_out, omp_out));
  ASSERT_GT(cpu_out.size(), 0u);
  if (meshify::available_cuda_devices() > 0) {
    auto cuda = meshify::create_grid_mesher(meshify::MeshBackend::Cuda, 0);
    ASSERT_NE(cuda, nullptr);
    const std::vector<float> gpu_out = run_backend(*cuda, cloud, req);
    ASSERT_TRUE(same_mesh(cpu_out, gpu_out));
  }
}

}  // namespace

TEST(FastGridMesh, BuildHasOpenMP)
{
  const std::string summary = meshify::build_summary();
  std::cout << "meshify build: " << summary
            << " cuda_devices=" << meshify::available_cuda_devices() << std::endl;
  EXPECT_NE(summary.find("openmp=on"), std::string::npos);
}

TEST(FastGridMesh, StrideMatchesPclLoop)
{
  EXPECT_EQ(meshify::effective_grid_stride(4, 0), 3);
  EXPECT_EQ(meshify::effective_grid_stride(1, 0), 1);
  EXPECT_EQ(meshify::effective_grid_stride(2, 0), 1);
  EXPECT_EQ(meshify::effective_grid_stride(4, 8), 8);
  const int sizes[] = {2, 3, 4, 5, 7, 8, 31, 48, 120};
  const int strides[] = {1, 2, 3, 4, 5, 7};
  for (int width : sizes) {
    for (int height : sizes) {
      for (int stride : strides) {
        const auto shape = meshify::make_grid_shape(width, height, stride);
        EXPECT_EQ(shape.ncells, brute_cells(width, height, stride))
          << width << "x" << height << " stride " << stride;
      }
    }
  }
  MeshRequest req;
  const MeshConfig cfg = meshify::make_mesh_config(16, 12, req);
  const float pcl_cos = std::abs(std::cos(pcl::deg2rad(12.5f)));
  EXPECT_FLOAT_EQ(cfg.cos_angle_tolerance, pcl_cos);
  EXPECT_EQ(cfg.stride, 3);
  EXPECT_EQ(cfg.max_edge_length_set, 0);
}

TEST(FastGridMesh, UnityPlaneNormal)
{
  const auto up = meshify::plane_from_unity_pose(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, false);
  ASSERT_TRUE(up.valid);
  EXPECT_NEAR(up.normal[0], 0.0, 1e-12);
  EXPECT_NEAR(up.normal[1], 1.0, 1e-12);
  EXPECT_NEAR(up.normal[2], 0.0, 1e-12);
  EXPECT_DOUBLE_EQ(up.point[0], 1.0);

  const auto flipped = meshify::plane_from_unity_pose(0, 0, 0, 0, 0, 0, 1, true);
  ASSERT_TRUE(flipped.valid);
  EXPECT_NEAR(flipped.normal[1], -1.0, 1e-12);

  constexpr double kPi = 3.141592653589793;
  const double s = std::sin(kPi / 4.0);
  const double c = std::cos(kPi / 4.0);
  const auto yaw = meshify::plane_from_unity_pose(0, 0, 0, 0, 0, s, c, false);
  ASSERT_TRUE(yaw.valid);
  EXPECT_NEAR(yaw.normal[0], -1.0, 1e-9);
  EXPECT_NEAR(yaw.normal[1], 0.0, 1e-9);
  EXPECT_NEAR(yaw.normal[2], 0.0, 1e-9);

  const auto scaled = meshify::plane_from_unity_pose(0, 0, 0, 0, 0, 0, 2, false);
  ASSERT_TRUE(scaled.valid);
  EXPECT_NEAR(scaled.normal[1], 1.0, 1e-12);

  const auto degenerate = meshify::plane_from_unity_pose(0, 0, 0, 0, 0, 0, 0, false);
  EXPECT_FALSE(degenerate.valid);
}

TEST(FastGridMesh, UnpackPointCloud2)
{
  const auto cloud = make_cloud(12, 8);
  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  PackedCloud unpacked;
  std::string error;
  ASSERT_TRUE(meshify::unpack_pointcloud2(msg, unpacked, error)) << error;
  ASSERT_EQ(unpacked.points.size(), cloud.points.size());
  EXPECT_TRUE(unpacked.has_color);
  int nan_points = 0;
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    EXPECT_TRUE(same_float(unpacked.points[i].x, cloud.points[i].x));
    EXPECT_TRUE(same_float(unpacked.points[i].y, cloud.points[i].y));
    EXPECT_TRUE(same_float(unpacked.points[i].z, cloud.points[i].z));
    if (std::isnan(cloud.points[i].x)) {
      ++nan_points;
    }
    EXPECT_EQ(unpacked.points[i].r, static_cast<float>(cloud.points[i].r) / 255.f);
    EXPECT_EQ(unpacked.points[i].g, static_cast<float>(cloud.points[i].g) / 255.f);
    EXPECT_EQ(unpacked.points[i].b, static_cast<float>(cloud.points[i].b) / 255.f);
  }
  EXPECT_GT(nan_points, 0);

  sensor_msgs::msg::PointCloud2 padded = msg;
  const uint32_t pad = 16;
  padded.row_step = msg.row_step + pad;
  padded.data.assign(static_cast<size_t>(padded.row_step) * padded.height, 0x5A);
  for (uint32_t y = 0; y < msg.height; ++y) {
    std::copy(
      msg.data.begin() + static_cast<size_t>(y) * msg.row_step,
      msg.data.begin() + static_cast<size_t>(y) * msg.row_step + msg.row_step,
      padded.data.begin() + static_cast<size_t>(y) * padded.row_step);
  }
  PackedCloud from_pad;
  ASSERT_TRUE(meshify::unpack_pointcloud2(padded, from_pad, error)) << error;
  ASSERT_EQ(from_pad.points.size(), unpacked.points.size());
  for (size_t i = 0; i < unpacked.points.size(); ++i) {
    EXPECT_TRUE(same_float(from_pad.points[i].x, unpacked.points[i].x));
    EXPECT_EQ(from_pad.points[i].r, unpacked.points[i].r);
  }

  sensor_msgs::msg::PointCloud2 flat = msg;
  flat.height = 1;
  flat.width = static_cast<uint32_t>(cloud.points.size());
  EXPECT_FALSE(meshify::unpack_pointcloud2(flat, unpacked, error));

  sensor_msgs::msg::PointCloud2 endian = msg;
  endian.is_bigendian = true;
  EXPECT_FALSE(meshify::unpack_pointcloud2(endian, unpacked, error));
}

TEST(FastGridMesh, MatchesLegacyOrganizedFastMesh)
{
  auto cpu = meshify::create_grid_mesher(meshify::MeshBackend::Cpu, 1);
  auto omp = meshify::create_grid_mesher(meshify::MeshBackend::OpenMp, 0);
  ASSERT_NE(cpu, nullptr);
  ASSERT_NE(omp, nullptr);

  const auto cloud = make_cloud(48, 36);
  expect_matches_pcl(*cpu, *omp, cloud, MeshRequest{}, true);

  sensor_msgs::msg::PointCloud2 msg;
  pcl::toROSMsg(cloud, msg);
  PackedCloud unpacked;
  std::string error;
  ASSERT_TRUE(meshify::unpack_pointcloud2(msg, unpacked, error)) << error;
  const MeshConfig cfg = meshify::make_mesh_config(unpacked.width, unpacked.height, MeshRequest{});
  std::vector<float> via_ros;
  cpu->mesh(unpacked, cfg, via_ros);
  const std::vector<float> direct = run_backend(*cpu, cloud, MeshRequest{});
  ASSERT_TRUE(same_mesh(direct, via_ros));
}

TEST(FastGridMesh, PlaneStrideEdgeAndShadowMatchPcl)
{
  auto cpu = meshify::create_grid_mesher(meshify::MeshBackend::Cpu, 1);
  auto omp = meshify::create_grid_mesher(meshify::MeshBackend::OpenMp, 2);
  ASSERT_NE(cpu, nullptr);
  ASSERT_NE(omp, nullptr);
  const auto cloud = make_cloud(40, 28);

  MeshRequest plane;
  plane.has_plane = true;
  plane.plane_ny = 1.0;
  plane.distance_threshold = 0.0;
  expect_matches_pcl(*cpu, *omp, cloud, plane, false);
  const auto kept = run_backend(*cpu, cloud, plane).size();
  const auto all = run_backend(*cpu, cloud, MeshRequest{}).size();
  EXPECT_LT(kept, all);

  // Shift the plane off the image center so +Y and -Y do not keep the same count.
  MeshRequest below = plane;
  below.plane_ny = -1.0;
  below.plane_py = -0.02;
  expect_matches_pcl(*cpu, *omp, cloud, below, false);
  EXPECT_NE(run_backend(*cpu, cloud, below).size(), kept);

  MeshRequest coarse;
  coarse.grid_stride = 4;
  expect_matches_pcl(*cpu, *omp, cloud, coarse, false);
  MeshRequest fine;
  fine.triangle_pixel_size = 1;
  expect_matches_pcl(*cpu, *omp, cloud, fine, false);
  EXPECT_GT(run_backend(*cpu, cloud, fine).size(), run_backend(*cpu, cloud, coarse).size());

  MeshRequest edges;
  edges.angle_tolerance_deg = -1.f;
  edges.max_edge_a = 0.02f;
  expect_matches_pcl(*cpu, *omp, cloud, edges, false);
  MeshRequest no_edge = edges;
  no_edge.max_edge_a = 0.f;
  EXPECT_LT(run_backend(*cpu, cloud, edges).size(), run_backend(*cpu, cloud, no_edge).size());

  MeshRequest depth = edges;
  depth.use_depth_as_distance = true;
  depth.max_edge_a = 0.05f;
  depth.max_edge_b = 0.01f;
  expect_matches_pcl(*cpu, *omp, cloud, depth, false);

  MeshRequest shadowed;
  shadowed.store_shadowed_faces = false;
  expect_matches_pcl(*cpu, *omp, cloud, shadowed, false);
  MeshRequest keep_shadow = shadowed;
  keep_shadow.store_shadowed_faces = true;
  expect_matches_pcl(*cpu, *omp, cloud, keep_shadow, false);
  EXPECT_GT(
    run_backend(*cpu, cloud, keep_shadow).size(),
    run_backend(*cpu, cloud, shadowed).size());

  MeshRequest shifted;
  shifted.viewpoint_x = 0.15f;
  shifted.viewpoint_y = -0.05f;
  shifted.viewpoint_z = 0.1f;
  expect_matches_pcl(*cpu, *omp, cloud, shifted, false);

  const auto wide = make_cloud(96, 64);
  expect_matches_pcl(*cpu, *omp, wide, MeshRequest{}, true);
}

TEST(FastGridMesh, CudaMatchesCpuWhenDevicePresent)
{
  if (meshify::available_cuda_devices() <= 0) {
    GTEST_SKIP() << "CUDA backend is not compiled or no GPU is visible. "
                 << "Build summary: " << meshify::build_summary();
  }
  auto cpu = meshify::create_grid_mesher(meshify::MeshBackend::Cpu, 1);
  auto omp = meshify::create_grid_mesher(meshify::MeshBackend::OpenMp, 0);
  const auto cloud = make_cloud(64, 40);
  expect_matches_pcl(*cpu, *omp, cloud, MeshRequest{}, true);
  MeshRequest plane;
  plane.has_plane = true;
  plane.distance_threshold = 0.0;
  plane.plane_ny = 1.0;
  expect_matches_pcl(*cpu, *omp, cloud, plane, false);
}
