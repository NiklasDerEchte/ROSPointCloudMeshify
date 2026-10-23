#include "meshify/fast_grid_mesh.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>

namespace meshify
{
namespace
{

using Steady = std::chrono::steady_clock;

int64_t us_between(Steady::time_point a, Steady::time_point b)
{
  return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
}

long long as_ll(int64_t v)
{
  return static_cast<long long>(v);
}

}  // namespace

class FastGridMeshNode : public rclcpp::Node
{
public:
  FastGridMeshNode()
  : rclcpp::Node("fast_grid_mesh_node")
  {
    input_topic_ = declare_parameter<std::string>("input_topic", "/points/xyzrgba");
    output_topic_ = declare_parameter<std::string>("output_topic", "/object_mesh/vertices");
    plane_topic_ = declare_parameter<std::string>("plane_topic", "/unity/clipping_plane");
    declare_parameter<std::string>("backend", "cpu");
    declare_parameter("triangle_pixel_size", 4);
    declare_parameter("grid_stride", 0);
    declare_parameter("distance_threshold", 0.01);
    declare_parameter("invert_plane_normal", false);
    declare_parameter("angle_tolerance_deg", 12.5);
    declare_parameter("max_edge_length_a", 0.0);
    declare_parameter("max_edge_length_b", 0.0);
    declare_parameter("max_edge_length_c", 0.0);
    declare_parameter("use_depth_as_distance", false);
    declare_parameter("store_shadowed_faces", false);
    declare_parameter("viewpoint_x", 0.0);
    declare_parameter("viewpoint_y", 0.0);
    declare_parameter("viewpoint_z", 0.0);
    declare_parameter("openmp_threads", 0);
    declare_parameter("log_every_n", 1);

    // Sensor streams: keep only the latest sample so a slow mesh cannot build a backlog.
    // Copy the QoS. keep_last() returns a reference to the temporary.
    const rclcpp::QoS input_qos{rclcpp::SensorDataQoS().keep_last(1)};
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, input_qos,
      std::bind(&FastGridMeshNode::on_cloud, this, std::placeholders::_1));

    rclcpp::QoS output_qos(rclcpp::KeepLast(1));
    output_qos.reliable();
    mesh_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(output_topic_, output_qos);

    rclcpp::QoS plane_qos(rclcpp::KeepLast(1));
    plane_qos.reliable();
    plane_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      plane_topic_, plane_qos,
      std::bind(&FastGridMeshNode::on_plane, this, std::placeholders::_1));

    ensure_backend(read_backend(), static_cast<int>(get_parameter("openmp_threads").as_int()));
    RCLCPP_INFO(
      get_logger(),
      "fast_grid_mesh_node ready backend=%s build(%s) input=%s output=%s plane=%s "
      "qos_in=best_effort/keep_last_1 qos_out=reliable/keep_last_1",
      mesher_->name(), build_summary(), input_topic_.c_str(), output_topic_.c_str(),
      plane_topic_.c_str());
  }

private:
  MeshBackend read_backend()
  {
    const std::string name = get_parameter("backend").as_string();
    MeshBackend backend;
    if (!parse_backend(name, backend)) {
      throw std::runtime_error(
              "unknown backend '" + name + "' (expected cpu, openmp, or cuda)");
    }
    if (!backend_compiled(backend)) {
      throw std::runtime_error(
              "backend '" + name + "' is not in this build (" + build_summary() +
              "). Install the CUDA toolkit and rebuild to enable cuda.");
    }
    return backend;
  }

  void ensure_backend(MeshBackend backend, int threads)
  {
    if (mesher_ && backend_ == backend && openmp_threads_ == threads) {
      return;
    }
    auto next = create_grid_mesher(backend, threads);
    if (!next) {
      throw std::runtime_error(
              std::string("failed to create backend ") + backend_name(backend));
    }
    mesher_ = std::move(next);
    backend_ = backend;
    openmp_threads_ = threads;
  }

  void on_plane(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    std::lock_guard<std::mutex> lock(plane_mutex_);
    plane_pose_ = msg->pose;
    has_plane_pose_ = true;
  }

  MeshRequest current_request()
  {
    MeshRequest req;
    req.triangle_pixel_size = static_cast<int>(get_parameter("triangle_pixel_size").as_int());
    req.grid_stride = static_cast<int>(get_parameter("grid_stride").as_int());
    req.angle_tolerance_deg = static_cast<float>(get_parameter("angle_tolerance_deg").as_double());
    req.max_edge_a = static_cast<float>(get_parameter("max_edge_length_a").as_double());
    req.max_edge_b = static_cast<float>(get_parameter("max_edge_length_b").as_double());
    req.max_edge_c = static_cast<float>(get_parameter("max_edge_length_c").as_double());
    req.use_depth_as_distance = get_parameter("use_depth_as_distance").as_bool();
    req.store_shadowed_faces = get_parameter("store_shadowed_faces").as_bool();
    req.viewpoint_x = static_cast<float>(get_parameter("viewpoint_x").as_double());
    req.viewpoint_y = static_cast<float>(get_parameter("viewpoint_y").as_double());
    req.viewpoint_z = static_cast<float>(get_parameter("viewpoint_z").as_double());
    req.distance_threshold = get_parameter("distance_threshold").as_double();

    geometry_msgs::msg::Pose pose;
    bool has_pose = false;
    {
      std::lock_guard<std::mutex> lock(plane_mutex_);
      has_pose = has_plane_pose_;
      pose = plane_pose_;
    }
    if (!has_pose) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "No Unity plane pose on %s yet — skip plane filter", plane_topic_.c_str());
      return req;
    }
    const bool invert = get_parameter("invert_plane_normal").as_bool();
    const PlaneModel plane = plane_from_unity_pose(
      pose.position.x, pose.position.y, pose.position.z,
      pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w,
      invert);
    if (!plane.valid) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Unity plane pose on %s is degenerate — skip plane filter", plane_topic_.c_str());
      return req;
    }
    req.has_plane = true;
    req.plane_px = plane.point[0];
    req.plane_py = plane.point[1];
    req.plane_pz = plane.point[2];
    req.plane_nx = plane.normal[0];
    req.plane_ny = plane.normal[1];
    req.plane_nz = plane.normal[2];
    return req;
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    const auto t_begin = Steady::now();
    try {
      ensure_backend(read_backend(), static_cast<int>(get_parameter("openmp_threads").as_int()));
      const auto t_des0 = Steady::now();
      std::string error;
      if (!unpack_pointcloud2(*msg, packed_, error)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "skip cloud: %s", error.c_str());
        return;
      }
      const auto t_des1 = Steady::now();

      const MeshRequest req = current_request();
      const MeshConfig cfg = make_mesh_config(packed_.width, packed_.height, req);
      const StageMicros stages = mesher_->mesh(packed_, cfg, wire_);
      const auto t_ser0 = Steady::now();

      std_msgs::msg::Float32MultiArray out_msg;
      out_msg.data.swap(wire_);
      const bool has_mesh = !out_msg.data.empty();
      const auto t_ser1 = Steady::now();
      if (has_mesh) {
        mesh_pub_->publish(out_msg);
      }
      const auto t_pub1 = Steady::now();
      wire_.swap(out_msg.data);
      out_msg.data.clear();

      const int64_t deserialize_us = us_between(t_des0, t_des1);
      const int64_t serialize_us = us_between(t_ser0, t_ser1);
      const int64_t publish_us = has_mesh ? us_between(t_ser1, t_pub1) : 0;
      const int64_t total_us = us_between(t_begin, t_pub1);
      const int triangles = static_cast<int>(wire_.size() / 21);

      ++frame_;
      int every = static_cast<int>(get_parameter("log_every_n").as_int());
      if (every < 1) {
        every = 1;
      }
      if (frame_ == 1 || frame_ % static_cast<uint64_t>(every) == 0) {
        RCLCPP_INFO(
          get_logger(),
          "MESHIFY_TIMING backend=%s %dx%d stride=%d points=%zu triangles=%d bytes=%zu "
          "deserialize_us=%lld upload_us=%lld clip_mesh_us=%lld compact_us=%lld "
          "download_us=%lld serialize_us=%lld publish_us=%lld total_us=%lld plane=%d",
          mesher_->name(), packed_.width, packed_.height, cfg.stride, packed_.points.size(),
          triangles, wire_.size() * sizeof(float),
          as_ll(deserialize_us), as_ll(stages.upload_us), as_ll(stages.clip_mesh_us),
          as_ll(stages.compact_us), as_ll(stages.download_us), as_ll(serialize_us),
          as_ll(publish_us), as_ll(total_us), req.has_plane ? 1 : 0);
      }
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(get_logger(), "fast grid mesh failed: %s", ex.what());
    }
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr plane_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mesh_pub_;

  std::string input_topic_;
  std::string output_topic_;
  std::string plane_topic_;

  std::mutex plane_mutex_;
  bool has_plane_pose_ = false;
  geometry_msgs::msg::Pose plane_pose_;

  PackedCloud packed_;
  std::vector<float> wire_;
  std::unique_ptr<GridMesher> mesher_;
  MeshBackend backend_ = MeshBackend::Cpu;
  int openmp_threads_ = 0;
  uint64_t frame_ = 0;
};

}  // namespace meshify

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<meshify::FastGridMeshNode>();
    rclcpp::spin(node);
  } catch (const std::exception & ex) {
    RCLCPP_FATAL(rclcpp::get_logger("fast_grid_mesh_node"), "%s", ex.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
