#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace meshify
{

/// One organized point after a single unpack of the PointCloud2.
/// Color is already the legacy wire scale (channel / 255). Alpha is not stored:
/// the wire format always writes 1, matching float_array_unity_plane_node.
struct PackedPoint
{
  float x = 0.f;
  float y = 0.f;
  float z = 0.f;
  float r = 0.f;
  float g = 1.f;
  float b = 0.f;
};

/// Legacy Float32MultiArray payload for one triangle: 3 * (x,y,z,r,g,b,a).
/// Binary-compatible with 21 floats so a later encoder can read this record
/// without another copy of the cloud. The compact format itself is out of scope.
struct TriangleRec
{
  float f[21];
};

struct PackedCloud
{
  int width = 0;
  int height = 0;
  bool has_color = false;
  std::vector<PackedPoint> points;
};

/// User-facing knobs. Defaults match the OrganizedFastMesh calls in
/// float_array_unity_plane_node (pixel size 4, right cut, PCL defaults).
struct MeshRequest
{
  int triangle_pixel_size = 4;
  /// 0 = derive the stride the way PCL does (max(1, triangle_pixel_size - 1)).
  /// A positive value is the LOD step in pixels and overrides that mapping.
  int grid_stride = 0;
  /// PCL default is 12.5 degrees. <= 0 disables the shadow-angle test.
  float angle_tolerance_deg = 12.5f;
  float max_edge_a = 0.f;
  float max_edge_b = 0.f;
  float max_edge_c = 0.f;
  bool use_depth_as_distance = false;
  bool store_shadowed_faces = false;
  float viewpoint_x = 0.f;
  float viewpoint_y = 0.f;
  float viewpoint_z = 0.f;
  bool has_plane = false;
  double plane_px = 0.0;
  double plane_py = 0.0;
  double plane_pz = 0.0;
  double plane_nx = 0.0;
  double plane_ny = 1.0;
  double plane_nz = 0.0;
  double distance_threshold = 0.01;
};

/// POD passed to every backend, including the CUDA kernel.
struct MeshConfig
{
  int width = 0;
  int height = 0;
  int stride = 1;
  float viewpoint_x = 0.f;
  float viewpoint_y = 0.f;
  float viewpoint_z = 0.f;
  float cos_angle_tolerance = -1.f;
  float max_edge_a = 0.f;
  float max_edge_b = 0.f;
  float max_edge_c = 0.f;
  int max_edge_length_set = 0;
  int use_depth_as_distance = 0;
  int store_shadowed_faces = 0;
  int has_plane = 0;
  double plane_px = 0.0;
  double plane_py = 0.0;
  double plane_pz = 0.0;
  double plane_nx = 0.0;
  double plane_ny = 1.0;
  double plane_nz = 0.0;
  double distance_threshold = 0.01;
};

struct GridShape
{
  int stride = 0;
  int cols = 0;
  int rows = 0;
  int ncells = 0;
};

struct StageMicros
{
  int64_t upload_us = 0;
  int64_t clip_mesh_us = 0;
  int64_t compact_us = 0;
  int64_t download_us = 0;
};

struct PlaneModel
{
  bool valid = false;
  double point[3] = {0.0, 0.0, 0.0};
  double normal[3] = {0.0, 1.0, 0.0};
};

/// Same factor as pcl::deg2rad(float) in PCL 1.12 / 1.14 (angles.h).
inline constexpr float kPclDeg2Rad = 0.017453293f;

inline int effective_grid_stride(int triangle_pixel_size, int grid_stride_override)
{
  if (grid_stride_override > 0) {
    return grid_stride_override;
  }
  const int pixel = triangle_pixel_size < 1 ? 1 : triangle_pixel_size;
  // OrganizedFastMesh::setTrianglePixelSize stores max(1, size - 1).
  // size 4 therefore walks the image with stride 3, not 4.
  const int stride = pixel - 1;
  return stride < 1 ? 1 : stride;
}

inline GridShape make_grid_shape(int width, int height, int stride)
{
  GridShape g;
  g.stride = stride;
  if (stride < 1 || width <= stride || height <= stride) {
    return g;
  }
  // Mirrors makeRightCutMesh: x = 0; x < width - stride; x += stride.
  const int last_col = width - stride;
  const int last_row = height - stride;
  g.cols = (last_col - 1) / stride + 1;
  g.rows = (last_row - 1) / stride + 1;
  g.ncells = g.cols * g.rows;
  return g;
}

inline MeshConfig make_mesh_config(int width, int height, const MeshRequest & req)
{
  MeshConfig cfg;
  cfg.width = width;
  cfg.height = height;
  cfg.stride = effective_grid_stride(req.triangle_pixel_size, req.grid_stride);
  cfg.viewpoint_x = req.viewpoint_x;
  cfg.viewpoint_y = req.viewpoint_y;
  cfg.viewpoint_z = req.viewpoint_z;
  if (req.angle_tolerance_deg > 0.f) {
    const float rad = req.angle_tolerance_deg * kPclDeg2Rad;
    cfg.cos_angle_tolerance = std::fabs(std::cos(rad));
  } else {
    cfg.cos_angle_tolerance = -1.f;
  }
  cfg.max_edge_a = req.max_edge_a;
  cfg.max_edge_b = req.max_edge_b;
  cfg.max_edge_c = req.max_edge_c;
  const float edge_sum = req.max_edge_a + req.max_edge_b + req.max_edge_c;
  cfg.max_edge_length_set =
    edge_sum > std::numeric_limits<float>::min() ? 1 : 0;
  cfg.use_depth_as_distance = req.use_depth_as_distance ? 1 : 0;
  cfg.store_shadowed_faces = req.store_shadowed_faces ? 1 : 0;
  cfg.has_plane = req.has_plane ? 1 : 0;
  cfg.plane_px = req.plane_px;
  cfg.plane_py = req.plane_py;
  cfg.plane_pz = req.plane_pz;
  cfg.plane_nx = req.plane_nx;
  cfg.plane_ny = req.plane_ny;
  cfg.plane_nz = req.plane_nz;
  cfg.distance_threshold = req.distance_threshold;
  return cfg;
}

static_assert(std::is_trivially_copyable<PackedPoint>::value, "PackedPoint must be a POD");
static_assert(std::is_trivially_copyable<TriangleRec>::value, "TriangleRec must be a POD");
static_assert(std::is_trivially_copyable<MeshConfig>::value, "MeshConfig must be a POD");
static_assert(sizeof(TriangleRec) == 21 * sizeof(float), "wire triangle is 21 packed floats");

}  // namespace meshify
