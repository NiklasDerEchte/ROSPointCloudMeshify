#pragma once

#include "meshify/mesh_types.hpp"

#include <cfloat>
#include <cmath>

#if defined(__CUDACC__)
#define MESHIFY_HD __host__ __device__
#else
#define MESHIFY_HD
#endif

namespace meshify
{

MESHIFY_HD inline float m_sqrt(float v)
{
#if defined(__CUDA_ARCH__)
  return sqrtf(v);
#else
  return std::sqrt(v);
#endif
}

MESHIFY_HD inline float m_fabs(float v)
{
#if defined(__CUDA_ARCH__)
  return fabsf(v);
#else
  return std::fabs(v);
#endif
}

MESHIFY_HD inline float m_max(float a, float b)
{
#if defined(__CUDA_ARCH__)
  return fmaxf(a, b);
#else
  return std::fmax(a, b);
#endif
}

MESHIFY_HD inline bool m_isnan(float v)
{
#if defined(__CUDA_ARCH__)
  return isnan(v);
#else
  return std::isnan(v);
#endif
}

MESHIFY_HD inline bool m_isfinite(float v)
{
#if defined(__CUDA_ARCH__)
  return isfinite(v);
#else
  return std::isfinite(v);
#endif
}

MESHIFY_HD inline bool point_usable(const PackedPoint & p, const MeshConfig & cfg)
{
  if (!m_isfinite(p.x) || !m_isfinite(p.y) || !m_isfinite(p.z)) {
    return false;
  }
  if (!cfg.has_plane) {
    return true;
  }
  // Same predicate as Float32ArrayPlaneNode::planeSegmentation:
  // signed distance = n · (p - plane_point), reject when <= threshold.
  // Coordinates are promoted to double, as in that node.
  const double dx = static_cast<double>(p.x) - cfg.plane_px;
  const double dy = static_cast<double>(p.y) - cfg.plane_py;
  const double dz = static_cast<double>(p.z) - cfg.plane_pz;
  const double sd = cfg.plane_nx * dx + cfg.plane_ny * dy + cfg.plane_nz * dz;
  return !(sd <= cfg.distance_threshold);
}

/// PCL OrganizedFastMesh::isShadowed, without the unused distance_tolerance
/// branch (the legacy node leaves it at the default, which disables it).
/// When max_edge_length is set, PCL replaces the angle result with the
/// edge-length test. That overwrite is kept.
MESHIFY_HD inline bool edge_rejected(
  const PackedPoint & a, const PackedPoint & b, const MeshConfig & cfg)
{
  const float dir_ax = cfg.viewpoint_x - a.x;
  const float dir_ay = cfg.viewpoint_y - a.y;
  const float dir_az = cfg.viewpoint_z - a.z;
  const float dir_bx = b.x - a.x;
  const float dir_by = b.y - a.y;
  const float dir_bz = b.z - a.z;
  const float distance_to_points = m_sqrt(dir_ax * dir_ax + dir_ay * dir_ay + dir_az * dir_az);
  const float distance_between_points = m_sqrt(dir_bx * dir_bx + dir_by * dir_by + dir_bz * dir_bz);

  bool valid = true;
  if (cfg.cos_angle_tolerance > 0.f) {
    float cos_angle =
      (dir_ax * dir_bx + dir_ay * dir_by + dir_az * dir_bz) /
      (distance_to_points * distance_between_points);
    if (m_isnan(cos_angle)) {
      cos_angle = 1.f;
    }
    const bool check_angle = m_fabs(cos_angle) >= cfg.cos_angle_tolerance;
    const bool check_distance = true;
    valid = !(check_angle && check_distance);
  }

  if (cfg.max_edge_length_set) {
    const float dist = cfg.use_depth_as_distance ? m_max(a.z, b.z) : distance_to_points;
    float dist_thresh = cfg.max_edge_a;
    if (m_fabs(cfg.max_edge_b) > FLT_MIN) {
      dist_thresh += cfg.max_edge_b * dist;
    }
    if (m_fabs(cfg.max_edge_c) > FLT_MIN) {
      dist_thresh += cfg.max_edge_c * dist * dist;
    }
    valid = distance_between_points <= dist_thresh;
  }
  return !valid;
}

MESHIFY_HD inline void write_vertex(float * d, const PackedPoint & p)
{
  d[0] = p.x;
  d[1] = p.y;
  d[2] = p.z;
  d[3] = p.r;
  d[4] = p.g;
  d[5] = p.b;
  d[6] = 1.f;
}

MESHIFY_HD inline void emit_one(
  const PackedPoint * points, const MeshConfig & cfg,
  int ia, int ib, int ic, TriangleRec & slot, uint8_t & flag)
{
  const PackedPoint & a = points[ia];
  const PackedPoint & b = points[ib];
  const PackedPoint & c = points[ic];
  if (!point_usable(a, cfg) || !point_usable(b, cfg) || !point_usable(c, cfg)) {
    flag = 0;
    return;
  }
  if (!cfg.store_shadowed_faces) {
    if (edge_rejected(a, b, cfg) || edge_rejected(b, c, cfg) || edge_rejected(c, a, cfg)) {
      flag = 0;
      return;
    }
  }
  write_vertex(slot.f + 0, a);
  write_vertex(slot.f + 7, b);
  write_vertex(slot.f + 14, c);
  flag = 1;
}

/// Right-cut of one quad, same vertex order as OrganizedFastMesh::makeRightCutMesh:
/// (origin, down-right, right) then (origin, down, down-right).
MESHIFY_HD inline void emit_cell(
  const PackedPoint * points, const MeshConfig & cfg,
  int x, int y, TriangleRec * slot2, uint8_t * flag2)
{
  const int stride = cfg.stride;
  const int i = y * cfg.width + x;
  const int i_right = i + stride;
  const int i_down = i + stride * cfg.width;
  const int i_down_right = i_down + stride;
  emit_one(points, cfg, i, i_down_right, i_right, slot2[0], flag2[0]);
  emit_one(points, cfg, i, i_down, i_down_right, slot2[1], flag2[1]);
}

}  // namespace meshify
