#include "meshify/fast_grid_mesh.hpp"

#include "meshify/grid_eval.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#if MESHIFY_HAS_OPENMP
#include <omp.h>
#endif

namespace meshify
{
namespace
{

using Clock = std::chrono::steady_clock;

int64_t us_between(Clock::time_point a, Clock::time_point b)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

float load_f32(const uint8_t * p)
{
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

uint32_t load_u32(const uint8_t * p)
{
  uint32_t v;
  std::memcpy(&v, p, sizeof(uint32_t));
  return v;
}

struct FieldRef
{
  int offset = -1;
  uint8_t datatype = 0;
  bool present() const {return offset >= 0;}
};

void decode_pcl_rgb(uint32_t packed, float & r, float & g, float & b)
{
  // PointXYZRGB stores the union as bytes b,g,r,a on little-endian.
  b = static_cast<float>(packed & 0xFFu) / 255.f;
  g = static_cast<float>((packed >> 8) & 0xFFu) / 255.f;
  r = static_cast<float>((packed >> 16) & 0xFFu) / 255.f;
}

enum class ColorMode
{
  None,
  Packed,
  Channels
};

void compact_slots(
  bool parallel, int threads,
  const std::vector<uint8_t> & flags, const std::vector<TriangleRec> & slots,
  std::vector<float> & out)
{
  const int n = static_cast<int>(flags.size());
#if MESHIFY_HAS_OPENMP
  if (parallel && n > 0) {
    int t = threads > 0 ? threads : omp_get_max_threads();
    if (t < 1) {
      t = 1;
    }
    std::vector<int> counts(static_cast<size_t>(t), 0);
    const int chunk = (n + t - 1) / t;
#pragma omp parallel num_threads(t)
    {
      const int tid = omp_get_thread_num();
      const int begin = tid * chunk;
      const int end = std::min(n, begin + chunk);
      int c = 0;
      for (int i = begin; i < end; ++i) {
        c += flags[static_cast<size_t>(i)] ? 1 : 0;
      }
      counts[static_cast<size_t>(tid)] = c;
    }
    std::vector<int> offsets(static_cast<size_t>(t) + 1, 0);
    for (int i = 0; i < t; ++i) {
      offsets[static_cast<size_t>(i) + 1] = offsets[static_cast<size_t>(i)] + counts[static_cast<size_t>(i)];
    }
    out.resize(static_cast<size_t>(offsets[static_cast<size_t>(t)]) * 21);
#pragma omp parallel num_threads(t)
    {
      const int tid = omp_get_thread_num();
      const int begin = tid * chunk;
      const int end = std::min(n, begin + chunk);
      size_t w = static_cast<size_t>(offsets[static_cast<size_t>(tid)]) * 21;
      for (int i = begin; i < end; ++i) {
        if (!flags[static_cast<size_t>(i)]) {
          continue;
        }
        std::memcpy(out.data() + w, slots[static_cast<size_t>(i)].f, 21 * sizeof(float));
        w += 21;
      }
    }
    return;
  }
#else
  (void)parallel;
  (void)threads;
#endif
  int count = 0;
  for (int i = 0; i < n; ++i) {
    count += flags[static_cast<size_t>(i)] ? 1 : 0;
  }
  out.resize(static_cast<size_t>(count) * 21);
  size_t w = 0;
  for (int i = 0; i < n; ++i) {
    if (!flags[static_cast<size_t>(i)]) {
      continue;
    }
    std::memcpy(out.data() + w, slots[static_cast<size_t>(i)].f, 21 * sizeof(float));
    w += 21;
  }
}

class CpuGridMesher : public GridMesher
{
public:
  CpuGridMesher(bool parallel, int threads)
  : parallel_(parallel), threads_(threads) {}

  const char * name() const override
  {
    return parallel_ ? "openmp" : "cpu";
  }

  StageMicros mesh(
    const PackedCloud & cloud, const MeshConfig & cfg, std::vector<float> & xyzrgba) override
  {
    if (cfg.width != cloud.width || cfg.height != cloud.height) {
      throw std::runtime_error("mesh config dimensions do not match the packed cloud");
    }
    const size_t expect =
      static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height);
    if (cloud.points.size() != expect) {
      throw std::runtime_error("packed cloud size does not match width*height");
    }

    const GridShape grid = make_grid_shape(cloud.width, cloud.height, cfg.stride);
    const int nslots = grid.ncells * 2;
    flags_.assign(static_cast<size_t>(nslots), 0);
    if (static_cast<int>(slots_.size()) < nslots) {
      slots_.resize(static_cast<size_t>(nslots));
    }

    StageMicros times;
    const auto t_upload0 = Clock::now();
    const auto t_upload1 = t_upload0;
    times.upload_us = 0;

    const PackedPoint * pts = cloud.points.empty() ? nullptr : cloud.points.data();
#if MESHIFY_HAS_OPENMP
    int threads = threads_ > 0 ? threads_ : omp_get_max_threads();
    if (threads < 1) {
      threads = 1;
    }
#pragma omp parallel for schedule(static) num_threads(threads) if(parallel_)
#endif
    for (int cell = 0; cell < grid.ncells; ++cell) {
      const int row = cell / grid.cols;
      const int col = cell - row * grid.cols;
      emit_cell(
        pts, cfg, col * cfg.stride, row * cfg.stride,
        &slots_[static_cast<size_t>(cell) * 2],
        &flags_[static_cast<size_t>(cell) * 2]);
    }
    const auto t_mesh1 = Clock::now();
    times.clip_mesh_us = us_between(t_upload1, t_mesh1);

    compact_slots(parallel_, threads_, flags_, slots_, xyzrgba);
    const auto t_compact1 = Clock::now();
    times.compact_us = us_between(t_mesh1, t_compact1);
    times.download_us = 0;
    return times;
  }

private:
  bool parallel_;
  int threads_;
  std::vector<uint8_t> flags_;
  std::vector<TriangleRec> slots_;
};

FieldRef find_field(const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
{
  FieldRef ref;
  for (const auto & field : msg.fields) {
    if (field.name == name) {
      ref.offset = static_cast<int>(field.offset);
      ref.datatype = field.datatype;
      break;
    }
  }
  return ref;
}

}  // namespace

bool parse_backend(const std::string & name, MeshBackend & out)
{
  if (name == "cpu") {
    out = MeshBackend::Cpu;
    return true;
  }
  if (name == "openmp") {
    out = MeshBackend::OpenMp;
    return true;
  }
  if (name == "cuda") {
    out = MeshBackend::Cuda;
    return true;
  }
  return false;
}

const char * backend_name(MeshBackend backend)
{
  switch (backend) {
    case MeshBackend::Cpu:
      return "cpu";
    case MeshBackend::OpenMp:
      return "openmp";
    case MeshBackend::Cuda:
      return "cuda";
  }
  return "unknown";
}

bool backend_compiled(MeshBackend backend)
{
  switch (backend) {
    case MeshBackend::Cpu:
      return true;
    case MeshBackend::OpenMp:
#if MESHIFY_HAS_OPENMP
      return true;
#else
      return false;
#endif
    case MeshBackend::Cuda:
#if MESHIFY_HAS_CUDA
      return true;
#else
      return false;
#endif
  }
  return false;
}

const char * build_summary()
{
#if MESHIFY_HAS_CUDA && MESHIFY_HAS_OPENMP
  return "openmp=on cuda=on";
#elif MESHIFY_HAS_CUDA
  return "openmp=off cuda=on";
#elif MESHIFY_HAS_OPENMP
  return "openmp=on cuda=off";
#else
  return "openmp=off cuda=off";
#endif
}

#if !MESHIFY_HAS_CUDA
int available_cuda_devices()
{
  return 0;
}
#endif

#if MESHIFY_HAS_CUDA
std::unique_ptr<GridMesher> create_cuda_grid_mesher();
#endif

std::unique_ptr<GridMesher> create_grid_mesher(MeshBackend backend, int openmp_threads)
{
  switch (backend) {
    case MeshBackend::Cpu:
      return std::make_unique<CpuGridMesher>(false, 1);
    case MeshBackend::OpenMp:
      return std::make_unique<CpuGridMesher>(true, openmp_threads);
    case MeshBackend::Cuda:
#if MESHIFY_HAS_CUDA
      return create_cuda_grid_mesher();
#else
      return nullptr;
#endif
  }
  return nullptr;
}

bool unpack_pointcloud2(
  const sensor_msgs::msg::PointCloud2 & msg, PackedCloud & out, std::string & error)
{
  if (msg.is_bigendian) {
    error = "big-endian PointCloud2 is not supported";
    return false;
  }
  if (msg.height < 2 || msg.width < 2) {
    error = "cloud is not organized (need width>=2 and height>=2)";
    return false;
  }
  if (msg.point_step < 12 || msg.row_step < msg.width * msg.point_step) {
    error = "point_step/row_step too small for the declared width";
    return false;
  }
  const size_t need =
    static_cast<size_t>(msg.row_step) * (msg.height - 1) +
    static_cast<size_t>(msg.width) * msg.point_step;
  if (msg.data.size() < need) {
    error = "PointCloud2 data is shorter than width*height";
    return false;
  }

  const FieldRef fx = find_field(msg, "x");
  const FieldRef fy = find_field(msg, "y");
  const FieldRef fz = find_field(msg, "z");
  if (!fx.present() || !fy.present() || !fz.present()) {
    error = "PointCloud2 is missing x, y, or z";
    return false;
  }
  if (fx.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    fy.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
    fz.datatype != sensor_msgs::msg::PointField::FLOAT32)
  {
    error = "x, y, z must be float32";
    return false;
  }
  const auto fits = [&](const FieldRef & field, uint32_t nbytes) {
      return field.present() &&
             static_cast<uint32_t>(field.offset) + nbytes <= msg.point_step;
    };
  if (!fits(fx, 4) || !fits(fy, 4) || !fits(fz, 4)) {
    error = "xyz field offset extends past point_step";
    return false;
  }

  const FieldRef fr = find_field(msg, "r");
  const FieldRef fg = find_field(msg, "g");
  const FieldRef fb = find_field(msg, "b");
  const FieldRef rgba = find_field(msg, "rgba");
  const FieldRef rgb = find_field(msg, "rgb");

  ColorMode mode = ColorMode::None;
  int packed_off = -1;
  int r_off = -1;
  int g_off = -1;
  int b_off = -1;
  if (fr.present() && fg.present() && fb.present() &&
    fr.datatype == sensor_msgs::msg::PointField::UINT8 &&
    fg.datatype == sensor_msgs::msg::PointField::UINT8 &&
    fb.datatype == sensor_msgs::msg::PointField::UINT8 &&
    fits(fr, 1) && fits(fg, 1) && fits(fb, 1))
  {
    mode = ColorMode::Channels;
    r_off = fr.offset;
    g_off = fg.offset;
    b_off = fb.offset;
  } else if (rgba.present() && fits(rgba, 4) &&
    (rgba.datatype == sensor_msgs::msg::PointField::FLOAT32 ||
    rgba.datatype == sensor_msgs::msg::PointField::UINT32))
  {
    mode = ColorMode::Packed;
    packed_off = rgba.offset;
  } else if (rgb.present() && fits(rgb, 4) &&
    (rgb.datatype == sensor_msgs::msg::PointField::FLOAT32 ||
    rgb.datatype == sensor_msgs::msg::PointField::UINT32))
  {
    mode = ColorMode::Packed;
    packed_off = rgb.offset;
  }

  out.width = static_cast<int>(msg.width);
  out.height = static_cast<int>(msg.height);
  out.has_color = mode != ColorMode::None;
  out.points.resize(static_cast<size_t>(out.width) * static_cast<size_t>(out.height));

  for (uint32_t y = 0; y < msg.height; ++y) {
    const uint8_t * row = msg.data.data() + static_cast<size_t>(y) * msg.row_step;
    for (uint32_t x = 0; x < msg.width; ++x) {
      const uint8_t * p = row + static_cast<size_t>(x) * msg.point_step;
      PackedPoint & dst = out.points[static_cast<size_t>(y) * msg.width + x];
      dst.x = load_f32(p + fx.offset);
      dst.y = load_f32(p + fy.offset);
      dst.z = load_f32(p + fz.offset);
      if (mode == ColorMode::Packed) {
        decode_pcl_rgb(load_u32(p + packed_off), dst.r, dst.g, dst.b);
      } else if (mode == ColorMode::Channels) {
        dst.r = static_cast<float>(p[r_off]) / 255.f;
        dst.g = static_cast<float>(p[g_off]) / 255.f;
        dst.b = static_cast<float>(p[b_off]) / 255.f;
      } else {
        dst.r = 0.f;
        dst.g = 1.f;
        dst.b = 0.f;
      }
    }
  }
  error.clear();
  return true;
}

PlaneModel plane_from_unity_pose(
  double px, double py, double pz,
  double qx, double qy, double qz, double qw,
  bool invert)
{
  PlaneModel model;
  model.point[0] = px;
  model.point[1] = py;
  model.point[2] = pz;

  Eigen::Quaterniond q(qw, qx, qy, qz);
  const double qn = q.norm();
  if (!std::isfinite(qn) || qn < 1e-12) {
    return model;
  }
  q.normalize();
  Eigen::Vector3d normal = q * Eigen::Vector3d::UnitY();
  const double nn = normal.norm();
  if (!std::isfinite(nn) || nn < 1e-9) {
    return model;
  }
  normal.normalize();
  if (invert) {
    normal = -normal;
  }
  model.normal[0] = normal.x();
  model.normal[1] = normal.y();
  model.normal[2] = normal.z();
  model.valid = true;
  return model;
}

}  // namespace meshify
