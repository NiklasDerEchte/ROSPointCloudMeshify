#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "meshify/float_array_cuda.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

class Float32ArrayCudaNode : public rclcpp::Node {
public:
    Float32ArrayCudaNode()
        : Node("float_array_cuda_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices"))
    {
        // Depth 1: a slow mesh must not queue a backlog of clouds.
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba", 1,
            std::bind(&Float32ArrayCudaNode::pointCloudCallback, this, std::placeholders::_1)
        );

        mesh_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(this->output_topic_, 1);
        meshify::initFloatArrayCuda();
        RCLCPP_INFO(this->get_logger(), "Node initialized, publishing Float32MultiArray on %s", output_topic_.c_str());
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        try {
            RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");

            bool hasRgb = false;
            bool hasRgba = false;
            for (const auto & field : msg->fields) {
                if (field.name == "rgb") {
                    hasRgb = true;
                    break;
                } else if (field.name == "rgba") {
                    hasRgba = true;
                    break;
                }
            }
            if (hasRgb || hasRgba) {
                RCLCPP_INFO(this->get_logger(), "Handle XYZ-RGB(A) PointCloud");
            } else {
                RCLCPP_INFO(this->get_logger(), "Handle XYZ PointCloud");
            }

            if (msg->height < 2 || msg->width < 2) {
                RCLCPP_WARN(this->get_logger(), "PointCloud is not organized, skip fast mesh");
                return;
            }

            const long t0 = nowMs();
            const long t_unpack = nowMs();
            if (!unpackPointCloud(msg)) {
                return;
            }
            const long unpack_ms = nowMs() - t_unpack;
            RCLCPP_INFO(this->get_logger(), "Unpack point cloud in %ld ms", unpack_ms);

            meshify::CudaMeshTimes times{};
            std::size_t nfinite = 0;
            const meshify::CudaMeshStatus status = meshify::runFloatArrayCuda(
                xyzrgb_.data(), width_, height_, distance_threshold_, vertex_array_.data, times, nfinite);

            RCLCPP_INFO(this->get_logger(), "Plane RANSAC on %zu/%zu finite points (%u x %u)",
                nfinite, xyzrgb_.size() / 6, static_cast<unsigned>(width_), static_cast<unsigned>(height_));

            if (status == meshify::CudaMeshStatus::NotEnoughFinite) {
                RCLCPP_WARN(this->get_logger(), "Not enough finite points for plane segmentation");
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                RCLCPP_INFO(this->get_logger(), "Total in %ld ms", nowMs() - t0);
                return;
            }
            if (status == meshify::CudaMeshStatus::NoPlane) {
                RCLCPP_WARN(this->get_logger(), "No plane found in the point cloud");
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                RCLCPP_INFO(this->get_logger(), "Total in %ld ms", nowMs() - t0);
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Plane segmentation in %ld ms", times.ransac_ms);
            RCLCPP_INFO(this->get_logger(), "Create organized triangulation mesh");
            RCLCPP_INFO(this->get_logger(), "Organized mesh created in %ld ms", times.mesh_ms);

            if (status == meshify::CudaMeshStatus::EmptyMesh || vertex_array_.data.empty()) {
                RCLCPP_WARN(this->get_logger(), "Triangulation failed: no polygons created");
                RCLCPP_WARN(this->get_logger(), "Float32MultiArray empty, skip publish");
                RCLCPP_INFO(this->get_logger(), "Total in %ld ms", nowMs() - t0);
                return;
            }

            RCLCPP_INFO(this->get_logger(), "publish %zu floats (%zu vertices)",
                vertex_array_.data.size(), vertex_array_.data.size() / 7);
            mesh_pub_->publish(vertex_array_);
            RCLCPP_INFO(this->get_logger(), "Total in %ld ms", nowMs() - t0);
            RCLCPP_INFO(this->get_logger(), "--- frame done ---");
        } catch (const std::exception & e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing PointCloud: %s", e.what());
        }
    }

    // ----------------------------------------------------------------------- [Unpack PointCloud2]

    // One read of the organized image into xyzrgb_ (x,y,z,r,g,b). No PCL copy.
    bool unpackPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & msg) {
        if (msg->is_bigendian) {
            RCLCPP_WARN(this->get_logger(), "Big-endian PointCloud2 is not supported");
            return false;
        }
        if (msg->point_step < 12 || msg->row_step < msg->width * msg->point_step) {
            RCLCPP_WARN(this->get_logger(), "PointCloud2 point_step/row_step is too small");
            return false;
        }
        const size_t need = static_cast<size_t>(msg->row_step) * (msg->height - 1) +
                            static_cast<size_t>(msg->width) * msg->point_step;
        if (msg->data.size() < need) {
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
        if (packed < 0) {
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
        for (int y = 0; y < height; ++y) {
            const uint8_t * row = base + static_cast<size_t>(y) * row_step;
            float * out = dst + static_cast<size_t>(y) * static_cast<size_t>(width) * 6;
            for (int x = 0; x < width; ++x) {
                const uint8_t * p = row + static_cast<size_t>(x) * point_step;
                float * q = out + static_cast<size_t>(x) * 6;
                q[0] = loadFloat(p + ox);
                q[1] = loadFloat(p + oy);
                q[2] = loadFloat(p + oz);
                if (channels) {
                    q[3] = static_cast<float>(p[o_r]) / 255.f;
                    q[4] = static_cast<float>(p[o_g]) / 255.f;
                    q[5] = static_cast<float>(p[o_b]) / 255.f;
                } else if (packed_ok) {
                    // PointXYZRGB stores the union as bytes b,g,r on little-endian.
                    const uint32_t c = loadUint(p + packed);
                    q[5] = static_cast<float>(c & 0xFFu) / 255.f;
                    q[4] = static_cast<float>((c >> 8) & 0xFFu) / 255.f;
                    q[3] = static_cast<float>((c >> 16) & 0xFFu) / 255.f;
                } else {
                    q[3] = 0.f;
                    q[4] = 1.f;
                    q[5] = 0.f;
                }
            }
        }
        return true;
    }

    static int findField(const sensor_msgs::msg::PointCloud2 & msg, const char * name, uint8_t & datatype) {
        for (const auto & field : msg.fields) {
            if (field.name == name) {
                datatype = field.datatype;
                return static_cast<int>(field.offset);
            }
        }
        datatype = 0;
        return -1;
    }

    static float loadFloat(const uint8_t * p) {
        float v;
        std::memcpy(&v, p, sizeof(float));
        return v;
    }

    static uint32_t loadUint(const uint8_t * p) {
        uint32_t v;
        std::memcpy(&v, p, sizeof(uint32_t));
        return v;
    }

    static long nowMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mesh_pub_;

    double distance_threshold_;
    std::string output_topic_;

    int width_{0};
    int height_{0};
    std::vector<float> xyzrgb_;
    std_msgs::msg::Float32MultiArray vertex_array_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<Float32ArrayCudaNode>();
        rclcpp::spin(node);
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("float_array_cuda_node"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
