#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include "meshify/float_array_cuda.hpp"
#include "meshify/frame_csv.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using meshify::elapsedMs;

/**
 * GPU (CUDA) mesh node, the counterpart of float_array_node in mode fast. Subscribes to an
 * organized PointCloud2, removes the dominant plane with RANSAC, builds the organized fast
 * mesh on the GPU and publishes the triangles as a Float32MultiArray
 * (TRIANGLE_LIST, 7 floats per vertex: x, y, z, r, g, b, a).
 */
class Float32ArrayCudaNode : public rclcpp::Node {
public:
    /// Declare parameters, create the CUDA context, open the CSV log and subscribe to /points/xyzrgba.
    Float32ArrayCudaNode()
        : Node("float_array_cuda_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices")),
          mode_(declare_parameter("mode", "fast")),
          enable_logging_(declare_parameter("enable_logging", true)),
          csv_path_(declare_parameter<std::string>("csv_path", "meshify_gpu.csv")),
          queue_depth_(declare_parameter("queue_depth", 100))
    {
        if (mode_ != "fast") {
            throw std::invalid_argument("mode must be fast, got '" + mode_ + "'");
        }
        meshify::initFloatArrayCuda();

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba",
            static_cast<std::size_t>(std::max<int64_t>(1, queue_depth_)),
            std::bind(&Float32ArrayCudaNode::pointCloudCallback, this, std::placeholders::_1));
        mesh_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(output_topic_, 10);
        RCLCPP_INFO(this->get_logger(), "Node initialized, publishing Float32MultiArray on %s", output_topic_.c_str());

        if (enable_logging_) {
            if (csv_path_.empty() || !meshify::openFrameCsv(csv_, csv_path_)) {
                RCLCPP_WARN(this->get_logger(),
                    "csv_path '%s' could not be opened or has a different header, CSV off", csv_path_.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "CSV frame log: %s", csv_path_.c_str());
            }
        }
        RCLCPP_INFO(this->get_logger(), "mode=%s queue_depth=%ld", mode_.c_str(), static_cast<long>(queue_depth_));
    }

private:
    static constexpr const char * kBackend = "gpu";

    // ----------------------------------------------------------------------- [Frame]

    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        const auto t_frame = std::chrono::steady_clock::now();
        const unsigned long long seq = ++frame_seq_;
        frame_.reset(seq);
        frame_.stamp_sec = msg->header.stamp.sec;
        frame_.stamp_nanosec = msg->header.stamp.nanosec;

        try {
            RCLCPP_INFO(this->get_logger(), "seq=%llu Received PointCloud2 message", seq);
            if (hasColor(*msg)) {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Handle XYZ-RGB(A) PointCloud", seq);
            } else {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Handle XYZ PointCloud", seq);
            }
            handlePointCloud(msg);
        } catch (const std::exception & e) {
            frame_.status = "error";
            RCLCPP_ERROR(this->get_logger(), "seq=%llu Error processing PointCloud: %s", seq, e.what());
        }

        frame_.total_ms = elapsedMs(t_frame);
        RCLCPP_INFO(this->get_logger(), "seq=%llu bytes %zu triangles %zu", seq, frame_.bytes, frame_.triangles);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Total in %.3f ms", seq, frame_.total_ms);
        meshify::writeFrameCsv(csv_, kBackend, frame_);
        RCLCPP_INFO(this->get_logger(), "seq=%llu --- frame done ---", seq);
    }

    void handlePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & msg) {
        const unsigned long long seq = frame_seq_;

        const auto t_unpack = std::chrono::steady_clock::now();
        unpackPointCloud(*msg);
        frame_.unpack_ms = elapsedMs(t_unpack);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Unpack point cloud in %.3f ms", seq, frame_.unpack_ms);

        // The organized fast mesh needs an image layout.
        if (height_ < 2 || width_ < 2) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu PointCloud is not organized, skip frame", seq);
            frame_.status = "not_organized";
            return;
        }

        // GPU: upload, plane segmentation, mesh and download in one call.
        meshify::CudaMeshTimes times{};
        std::size_t nfinite = 0;
        meshify::CudaMeshStatus status = meshify::CudaMeshStatus::Ok;
        try {
            status = meshify::runFloatArrayCuda(
                xyzrgb_.data(), width_, height_, distance_threshold_, vertex_array_.data, times, nfinite);
        } catch (...) {
            storeGpuTimes(times);
            throw;
        }
        storeGpuTimes(times);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Upload to GPU in %.3f ms", seq, frame_.upload_ms);

        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane RANSAC on %zu/%zu finite points (%u x %u)",
            seq, nfinite, xyzrgb_.size() / 6, static_cast<unsigned>(width_), static_cast<unsigned>(height_));
        if (status == meshify::CudaMeshStatus::NotEnoughFinite) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Not enough finite points for plane segmentation", seq);
            RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
            frame_.status = "not_enough_finite";
            return;
        }
        if (status == meshify::CudaMeshStatus::NoPlane) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu No plane found in the point cloud", seq);
            RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
            frame_.status = "no_plane";
            return;
        }
        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane segmentation in %.3f ms", seq, frame_.plane_ms);

        RCLCPP_INFO(this->get_logger(), "seq=%llu Create organized triangulation mesh", seq);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Organized mesh created in %.3f ms", seq, frame_.mesh_ms);

        RCLCPP_INFO(this->get_logger(), "seq=%llu Float32MultiArray payload in %.3f ms (%zu vertices)",
            seq, frame_.payload_ms, vertex_array_.data.size() / 7);
        if (vertex_array_.data.empty()) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Triangulation failed: no polygons created, skip publish", seq);
            frame_.status = "empty_mesh";
            return;
        }

        publishMesh();
    }

    void publishMesh() {
        frame_.bytes = vertex_array_.data.size() * sizeof(float);
        frame_.triangles = vertex_array_.data.size() / 21;
        const auto t_publish = std::chrono::steady_clock::now();
        mesh_pub_->publish(vertex_array_);
        frame_.publish_ms = elapsedMs(t_publish);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Publish %zu floats (%zu vertices) in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), vertex_array_.data.size(),
            vertex_array_.data.size() / 7, frame_.publish_ms);
    }

    static bool hasColor(const sensor_msgs::msg::PointCloud2 & msg) {
        return std::any_of(msg.fields.begin(), msg.fields.end(),
            [](const auto & field) { return field.name == "rgb" || field.name == "rgba"; });
    }

    // Keeps the stage times measured up to a failing CUDA call as well.
    void storeGpuTimes(const meshify::CudaMeshTimes & times) {
        frame_.upload_ms = times.upload_ms;
        frame_.plane_ms = times.plane_ms;
        frame_.mesh_ms = times.mesh_ms;
        frame_.payload_ms = times.payload_ms;
    }

    // ----------------------------------------------------------------------- [Unpack PointCloud2]

    // One pass over the image into xyzrgb_ (x, y, z, r, g, b with colors /255), the GPU input
    // layout. Clouds without color are green. Throws on a layout it cannot read, like pcl::fromROSMsg.
    void unpackPointCloud(const sensor_msgs::msg::PointCloud2 & msg) {
        if (msg.is_bigendian) {
            throw std::runtime_error("big-endian PointCloud2 is not supported");
        }
        if (msg.point_step < 12 || msg.row_step < msg.width * msg.point_step) {
            throw std::runtime_error("PointCloud2 point_step/row_step is too small");
        }
        const std::size_t need = msg.height == 0 ? 0 :
            static_cast<std::size_t>(msg.row_step) * (msg.height - 1) +
            static_cast<std::size_t>(msg.width) * msg.point_step;
        if (msg.data.size() < need) {
            throw std::runtime_error("PointCloud2 data is shorter than width*height");
        }

        uint8_t tx = 0, ty = 0, tz = 0;
        const int ox = findField(msg, "x", tx);
        const int oy = findField(msg, "y", ty);
        const int oz = findField(msg, "z", tz);
        if (ox < 0 || oy < 0 || oz < 0 ||
            tx != sensor_msgs::msg::PointField::FLOAT32 ||
            ty != sensor_msgs::msg::PointField::FLOAT32 ||
            tz != sensor_msgs::msg::PointField::FLOAT32)
        {
            throw std::runtime_error("PointCloud2 x/y/z must be float32");
        }

        uint8_t tr = 0, tg = 0, tb = 0, tpacked = 0;
        const int o_r = findField(msg, "r", tr);
        const int o_g = findField(msg, "g", tg);
        const int o_b = findField(msg, "b", tb);
        int packed = findField(msg, "rgba", tpacked);
        if (packed < 0) {
            packed = findField(msg, "rgb", tpacked);
        }
        const bool channels = o_r >= 0 && o_g >= 0 && o_b >= 0 &&
                              tr == sensor_msgs::msg::PointField::UINT8 &&
                              tg == sensor_msgs::msg::PointField::UINT8 &&
                              tb == sensor_msgs::msg::PointField::UINT8;
        const bool packed_ok = packed >= 0 &&
                               (tpacked == sensor_msgs::msg::PointField::FLOAT32 ||
                                tpacked == sensor_msgs::msg::PointField::UINT32);

        width_ = static_cast<int>(msg.width);
        height_ = static_cast<int>(msg.height);
        xyzrgb_.resize(static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 6);

        const int width = width_;
        const int height = height_;
        const uint32_t point_step = msg.point_step;
        const uint32_t row_step = msg.row_step;
        const uint8_t * base = msg.data.data();
        float * dst = xyzrgb_.data();

#pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y) {
            const uint8_t * row = base + static_cast<std::size_t>(y) * row_step;
            float * out = dst + static_cast<std::size_t>(y) * static_cast<std::size_t>(width) * 6;
            for (int x = 0; x < width; ++x) {
                const uint8_t * p = row + static_cast<std::size_t>(x) * point_step;
                float * q = out + static_cast<std::size_t>(x) * 6;
                q[0] = loadFloat(p + ox);
                q[1] = loadFloat(p + oy);
                q[2] = loadFloat(p + oz);
                if (channels) {
                    q[3] = static_cast<float>(p[o_r]) / 255.f;
                    q[4] = static_cast<float>(p[o_g]) / 255.f;
                    q[5] = static_cast<float>(p[o_b]) / 255.f;
                } else if (packed_ok) {
                    // PointXYZRGB stores the union as bytes b, g, r on little-endian.
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

    // ----------------------------------------------------------------------- [Members]

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mesh_pub_;

    double distance_threshold_;
    std::string output_topic_;
    std::string mode_;
    bool enable_logging_;
    std::string csv_path_;
    int64_t queue_depth_;

    // rclcpp::spin is single-threaded, so one frame owns these fields.
    uint64_t frame_seq_{0};
    meshify::FrameTimes frame_;
    std::ofstream csv_;
    std_msgs::msg::Float32MultiArray vertex_array_;

    // GPU input: the unpacked organized image.
    int width_{0};
    int height_{0};
    std::vector<float> xyzrgb_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<Float32ArrayCudaNode>());
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("float_array_cuda_node"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
