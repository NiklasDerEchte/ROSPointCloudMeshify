#pragma once

#include "meshify/mesh_types.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <memory>
#include <string>
#include <vector>

namespace meshify
{

class GridMesher
{
public:
  virtual ~GridMesher() = default;

  /// Clip, triangulate and compact into the legacy xyzrgba float buffer.
  /// `xyzrgba` is resized to 21 floats per kept triangle and reused across frames.
  virtual StageMicros mesh(
    const PackedCloud & cloud, const MeshConfig & cfg, std::vector<float> & xyzrgba) = 0;

  virtual const char * name() const = 0;
};

enum class MeshBackend
{
  Cpu,
  OpenMp,
  Cuda
};

bool parse_backend(const std::string & name, MeshBackend & out);
const char * backend_name(MeshBackend backend);
bool backend_compiled(MeshBackend backend);

std::unique_ptr<GridMesher> create_grid_mesher(MeshBackend backend, int openmp_threads);

/// 0 when the CUDA backend was not compiled or no device is visible.
int available_cuda_devices();

/// "openmp=on cuda=off" style summary of this binary.
const char * build_summary();

/// One pass over the PointCloud2 into a packed organized image.
/// Fails on unorganized clouds, big-endian data, or a missing xyz layout.
bool unpack_pointcloud2(
  const sensor_msgs::msg::PointCloud2 & msg, PackedCloud & out, std::string & error);

/// Unity clipping plane: local +Y rotated into the pose frame.
/// `invert` matches invert_plane_normal on the legacy node.
/// A degenerate quaternion yields valid=false (no clipping).
PlaneModel plane_from_unity_pose(
  double px, double py, double pz,
  double qx, double qy, double qz, double qw,
  bool invert);

}  // namespace meshify
