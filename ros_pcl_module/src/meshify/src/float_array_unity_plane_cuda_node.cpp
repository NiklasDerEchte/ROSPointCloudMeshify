#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <Eigen/Geometry>

#include "meshify/organized_fast_mesh_cuda.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

class Float32ArrayPlaneCudaNode : public rclcpp::Node
{
public:
    Float32ArrayPlaneCudaNode()
        : Node("float_array_unity_plane_cuda_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices")),
          plane_topic_(declare_parameter("plane_topic", "/unity/clipping_plane")),
          invert_plane_normal_(declare_parameter("invert_plane_normal", false))
    {
        // Depth 1: a slow mesh must not queue a backlog of clouds.
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points/xyzrgba", 1,
            std::bind(&Float32ArrayPlaneCudaNode::pointCloudCallback, this, std::placeholders::_1));

        plane_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            plane_topic_, 1,
            std::bind(&Float32ArrayPlaneCudaNode::planePoseCallback, this, std::placeholders::_1));

        mesh_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(this->output_topic_, 1);

        meshify::initOrganizedFastMeshCuda();
        RCLCPP_INFO(this->get_logger(),
                    "Node initialized: mesh=%s plane=%s (organized fast mesh, CUDA)",
                    output_topic_.c_str(), plane_topic_.c_str());
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try
        {
            RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");

            long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (!unpackPointCloud(msg))
            {
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Unpack point cloud in %ld ms",
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);

            meshify::UnityPlane plane{};
            plane.distance_threshold = distance_threshold_;
            Eigen::Vector3d plane_point;
            Eigen::Vector3d plane_normal;
            if (!getUnityPlane(plane_point, plane_normal))
            {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 2000,
                    "No Unity plane pose on %s yet — skip plane filter", plane_topic_.c_str());
            }
            else
            {
                plane.valid = 1;
                plane.px = plane_point.x();
                plane.py = plane_point.y();
                plane.pz = plane_point.z();
                plane.nx = plane_normal.x();
                plane.ny = plane_normal.y();
                plane.nz = plane_normal.z();
            }

            meshify::CudaMeshTimes times;
            meshify::runOrganizedFastMeshCuda(
                xyzrgb_.data(), width_, height_, plane, vertex_array_.data, times);
            RCLCPP_INFO(this->get_logger(), "Upload point cloud in %ld ms", times.upload_ms);
            RCLCPP_INFO(this->get_logger(), "Organized mesh created in %ld ms", times.mesh_ms);
            RCLCPP_INFO(this->get_logger(), "Compact triangles in %ld ms", times.compact_ms);
            RCLCPP_INFO(this->get_logger(), "Download mesh in %ld ms", times.download_ms);

            if (vertex_array_.data.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Float32MultiArray empty, skip publish");
                return;
            }

            t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            mesh_pub_->publish(vertex_array_);
            RCLCPP_INFO(this->get_logger(), "publish %zu floats (%zu vertices) in %ld ms",
                        vertex_array_.data.size(), vertex_array_.data.size() / 7,
                        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
            RCLCPP_INFO(this->get_logger(), "--- frame done ---");
        }
        catch (const std::exception & e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error processing PointCloud: %s", e.what());
        }
    }

    // ----------------------------------------------------------------------- [Unpack PointCloud2]

    // One read of the organized image into xyzrgb_ (x,y,z,r,g,b). No PCL copy.
    bool unpackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & msg)
    {
        if (msg->is_bigendian)
        {
            RCLCPP_WARN(this->get_logger(), "Big-endian PointCloud2 is not supported");
            return false;
        }
        if (msg->height < 2 || msg->width < 2)
        {
            RCLCPP_WARN(this->get_logger(), "PointCloud is not organized, skip fast mesh");
            return false;
        }
        if (msg->point_step < 12 || msg->row_step < msg->width * msg->point_step)
        {
            RCLCPP_WARN(this->get_logger(), "PointCloud2 point_step/row_step is too small");
            return false;
        }
        const size_t need = static_cast<size_t>(msg->row_step) * (msg->height - 1) +
                            static_cast<size_t>(msg->width) * msg->point_step;
        if (msg->data.size() < need)
        {
            RCLCPP_WARN(this->get_logger(), "PointCloud2 data is shorter than width*height");
            return false;
        }

        uint8_t tx = 0, ty = 0, tz = 0;
        const int ox = findField(*msg, "x", tx);
        const int oy = findField(*msg, "y", ty);
        const int oz = findField(*msg, "z", tz);
        if (ox < 0 || oy < 0 || oz < 0 ||
            tx != sensor_msgs::msg::PointField::FLOAT32 ||
            ty != sensor_msgs::msg::PointField::FLOAT32 ||
            tz != sensor_msgs::msg::PointField::FLOAT32)
        {
            RCLCPP_WARN(this->get_logger(), "PointCloud2 x/y/z must be float32");
            return false;
        }

        uint8_t tr = 0, tg = 0, tb = 0, tpacked = 0;
        const int o_r = findField(*msg, "r", tr);
        const int o_g = findField(*msg, "g", tg);
        const int o_b = findField(*msg, "b", tb);
        int packed = findField(*msg, "rgba", tpacked);
        if (packed < 0)
        {
            packed = findField(*msg, "rgb", tpacked);
        }
        const bool channels = o_r >= 0 && o_g >= 0 && o_b >= 0 &&
                              tr == sensor_msgs::msg::PointField::UINT8 &&
                              tg == sensor_msgs::msg::PointField::UINT8 &&
                              tb == sensor_msgs::msg::PointField::UINT8;
        const bool packed_ok = packed >= 0 &&
                               (tpacked == sensor_msgs::msg::PointField::FLOAT32 ||
                                tpacked == sensor_msgs::msg::PointField::UINT32);

        width_ = static_cast<int>(msg->width);
        height_ = static_cast<int>(msg->height);
        xyzrgb_.resize(static_cast<size_t>(width_) * static_cast<size_t>(height_) * 6);

        const int width = width_;
        const int height = height_;
        const uint32_t point_step = msg->point_step;
        const uint32_t row_step = msg->row_step;
        const uint8_t * base = msg->data.data();
        float * dst = xyzrgb_.data();

#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y)
        {
            const uint8_t * row = base + static_cast<size_t>(y) * row_step;
            float * out = dst + static_cast<size_t>(y) * static_cast<size_t>(width) * 6;
            for (int x = 0; x < width; ++x)
            {
                const uint8_t * p = row + static_cast<size_t>(x) * point_step;
                float * q = out + static_cast<size_t>(x) * 6;
                q[0] = loadFloat(p + ox);
                q[1] = loadFloat(p + oy);
                q[2] = loadFloat(p + oz);
                if (channels)
                {
                    q[3] = static_cast<float>(p[o_r]) / 255.f;
                    q[4] = static_cast<float>(p[o_g]) / 255.f;
                    q[5] = static_cast<float>(p[o_b]) / 255.f;
                }
                else if (packed_ok)
                {
                    // PointXYZRGB stores the union as bytes b,g,r on little-endian.
                    const uint32_t c = loadUint(p + packed);
                    q[5] = static_cast<float>(c & 0xFFu) / 255.f;
                    q[4] = static_cast<float>((c >> 8) & 0xFFu) / 255.f;
                    q[3] = static_cast<float>((c >> 16) & 0xFFu) / 255.f;
                }
                else
                {
                    q[3] = 0.f;
                    q[4] = 1.f;
                    q[5] = 0.f;
                }
            }
        }
        return true;
    }

    static int findField(const sensor_msgs::msg::PointCloud2 & msg, const char * name, uint8_t & datatype)
    {
        for (const auto & field : msg.fields)
        {
            if (field.name == name)
            {
                datatype = field.datatype;
                return static_cast<int>(field.offset);
            }
        }
        datatype = 0;
        return -1;
    }

    static float loadFloat(const uint8_t * p)
    {
        float v;
        std::memcpy(&v, p, sizeof(float));
        return v;
    }

    static uint32_t loadUint(const uint8_t * p)
    {
        uint32_t v;
        std::memcpy(&v, p, sizeof(uint32_t));
        return v;
    }

    // ----------------------------------------------------------------------- [Unity plane]

    void planePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (msg == nullptr)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(plane_mutex_);
        plane_pose_ = msg->pose;
        has_plane_pose_ = true;
    }

    /// Returns false if no Unity plane pose received yet.
    /// Normal = rotated Unity +Y. Signed distance = n · (p - plane_point).
    bool getUnityPlane(Eigen::Vector3d & point_on_plane, Eigen::Vector3d & normal) const
    {
        geometry_msgs::msg::Pose pose;
        {
            std::lock_guard<std::mutex> lock(plane_mutex_);
            if (!has_plane_pose_)
            {
                return false;
            }
            pose = plane_pose_;
        }

        point_on_plane = Eigen::Vector3d(
            pose.position.x,
            pose.position.y,
            pose.position.z);

        Eigen::Quaterniond q(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);
        q.normalize();
        normal = q * Eigen::Vector3d::UnitY();
        if (normal.norm() < 1e-9)
        {
            return false;
        }
        normal.normalize();
        if (invert_plane_normal_)
        {
            normal = -normal;
        }
        return true;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr plane_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mesh_pub_;

    mutable std::mutex plane_mutex_;
    bool has_plane_pose_{false};
    geometry_msgs::msg::Pose plane_pose_;

    double distance_threshold_;
    std::string output_topic_;
    std::string plane_topic_;
    bool invert_plane_normal_;

    int width_{0};
    int height_{0};
    std::vector<float> xyzrgb_;
    std_msgs::msg::Float32MultiArray vertex_array_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    try
    {
        auto node = std::make_shared<Float32ArrayPlaneCudaNode>();
        rclcpp::spin(node);
    }
    catch (const std::exception & e)
    {
        RCLCPP_FATAL(rclcpp::get_logger("float_array_unity_plane_cuda_node"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
