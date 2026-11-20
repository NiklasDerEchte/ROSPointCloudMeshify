#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/organized_fast_mesh.h>

#include "meshify/frame_csv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

using meshify::elapsedMs;

/**
 * CPU (PCL) mesh node, the counterpart of float_array_cuda_node. Subscribes to an organized
 * PointCloud2, removes the dominant plane with RANSAC, builds the organized fast mesh and
 * publishes the triangles as a Float32MultiArray (TRIANGLE_LIST, 7 floats per vertex:
 * x, y, z, r, g, b, a). Greedy and Poisson meshing are in float_array_pcl_node.
 */
class Float32ArrayNode : public rclcpp::Node {
public:
    /// Declare parameters, open the CSV log and subscribe to /points/xyzrgba.
    Float32ArrayNode()
        : Node("float_array_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices")),
          mode_(declare_parameter("mode", "fast")),
          enable_logging_(declare_parameter("enable_logging", true)),
          csv_path_(declare_parameter<std::string>("csv_path", "meshify_cpu.csv")),
          queue_depth_(declare_parameter("queue_depth", 100))
    {
        if (mode_ != "fast") {
            throw std::invalid_argument("mode must be fast, got '" + mode_ + "'");
        }

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba",
            static_cast<std::size_t>(std::max<int64_t>(1, queue_depth_)),
            std::bind(&Float32ArrayNode::pointCloudCallback, this, std::placeholders::_1));
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
    static constexpr const char * kBackend = "cpu";

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
                handlePointCloud<pcl::PointXYZRGB>(msg);
            } else {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Handle XYZ PointCloud", seq);
                handlePointCloud<pcl::PointXYZ>(msg);
            }
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

    template <typename PointT>
    void handlePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & msg) {
        const unsigned long long seq = frame_seq_;

        const auto t_unpack = std::chrono::steady_clock::now();
        typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*msg, *cloud);
        frame_.unpack_ms = elapsedMs(t_unpack);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Unpack point cloud in %.3f ms", seq, frame_.unpack_ms);

        // The organized fast mesh needs an image layout.
        if (cloud->height < 2 || cloud->width < 2) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu PointCloud is not organized, skip frame", seq);
            frame_.status = "not_organized";
            return;
        }

        typename pcl::PointCloud<PointT>::Ptr cloud_filtered = planeSegmentation<PointT>(cloud);
        if (cloud_filtered == nullptr) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
            return;
        }

        RCLCPP_INFO(this->get_logger(), "seq=%llu Create organized triangulation mesh", seq);
        pcl::PolygonMesh mesh;
        createOrganizedTriangulationMesh<PointT>(cloud_filtered, mesh);

        convertMeshToFloatArray<PointT>(mesh);
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

    // ----------------------------------------------------------------------- [Plane segmentation]

    // RANSAC plane on the finite points. The plane inliers are set to NaN, so the cloud stays
    // organized. Returns nullptr if there is no plane.
    template <typename PointT>
    typename pcl::PointCloud<PointT>::Ptr planeSegmentation(const typename pcl::PointCloud<PointT>::Ptr & cloud) {
        const unsigned long long seq = frame_seq_;

        // Segments `cloud` in place, like the GPU segments its device buffer: the unpacked
        // input is not needed afterwards, so there is no working copy.
        const std::size_t npoints = cloud->size();
        const unsigned width = cloud->width;
        const unsigned height = cloud->height;

        const auto t_plane = std::chrono::steady_clock::now();
        pcl::IndicesPtr finite(new pcl::Indices());
        finite->reserve(cloud->size());
        for (std::size_t i = 0; i < cloud->size(); ++i) {
            const auto & p = cloud->points[i];
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                finite->push_back(static_cast<int>(i));
            }
        }
        if (finite->size() < 3) {
            frame_.plane_ms = elapsedMs(t_plane);
            logRansacInput(finite->size(), npoints, width, height);
            RCLCPP_WARN(this->get_logger(), "seq=%llu Not enough finite points for plane segmentation", seq);
            frame_.status = "not_enough_finite";
            return nullptr;
        }

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<PointT> seg;
        seg.setOptimizeCoefficients(true);
        seg.setMaxIterations(1000);
        seg.setProbability(0.85);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold_);
        seg.setInputCloud(cloud);
        seg.setIndices(finite);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.empty()) {
            frame_.plane_ms = elapsedMs(t_plane);
            logRansacInput(finite->size(), npoints, width, height);
            RCLCPP_WARN(this->get_logger(), "seq=%llu No plane found in the point cloud", seq);
            frame_.status = "no_plane";
            return nullptr;
        }

        constexpr float nan = std::numeric_limits<float>::quiet_NaN();
        for (const int idx : inliers->indices) {
            cloud->points[idx].x = nan;
            cloud->points[idx].y = nan;
            cloud->points[idx].z = nan;
        }
        frame_.plane_ms = elapsedMs(t_plane);
        logRansacInput(finite->size(), npoints, width, height);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane segmentation in %.3f ms", seq, frame_.plane_ms);
        return cloud;
    }

    void logRansacInput(std::size_t nfinite, std::size_t npoints, unsigned width, unsigned height) {
        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane RANSAC on %zu/%zu finite points (%u x %u)",
            static_cast<unsigned long long>(frame_seq_), nfinite, npoints, width, height);
    }

    // ----------------------------------------------------------------------- [Mesh]

    template <typename PointT>
    void createOrganizedTriangulationMesh(const typename pcl::PointCloud<PointT>::Ptr & cloud, pcl::PolygonMesh & mesh) {
        const auto t_mesh = std::chrono::steady_clock::now();
        pcl::OrganizedFastMesh<PointT> ofm;
        ofm.setInputCloud(cloud);
        ofm.setTrianglePixelSize(4);
        ofm.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
        ofm.reconstruct(mesh);
        frame_.mesh_ms = elapsedMs(t_mesh);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Organized mesh created in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), frame_.mesh_ms);
    }

    // ----------------------------------------------------------------------- [Mesh -> Float32MultiArray]

    // Fills vertex_array_ with the triangles of `mesh`, 7 floats per vertex (x, y, z, r, g, b, a).
    // Clouds without color are drawn green.
    template <typename PointT>
    void convertMeshToFloatArray(const pcl::PolygonMesh & mesh) {
        const auto t_convert = std::chrono::steady_clock::now();
        pcl::PointCloud<PointT> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        std::size_t tri = 0;
        for (const auto & polygon : mesh.polygons) {
            if (polygon.vertices.size() == 3) {
                ++tri;
            }
        }
        vertex_array_.data.resize(tri * 21);
        float * d = vertex_array_.data.data();
        for (const auto & polygon : mesh.polygons) {
            if (polygon.vertices.size() != 3) {
                continue;
            }
            for (const auto vertex_idx : polygon.vertices) {
                const auto & point = cloud.points[vertex_idx];
                d[0] = point.x;
                d[1] = point.y;
                d[2] = point.z;
                if constexpr (std::is_same_v<PointT, pcl::PointXYZRGB>) {
                    d[3] = point.r / 255.0f;
                    d[4] = point.g / 255.0f;
                    d[5] = point.b / 255.0f;
                } else {
                    d[3] = 0.0f;
                    d[4] = 1.0f;
                    d[5] = 0.0f;
                }
                d[6] = 1.0f;
                d += 7;
            }
        }
        frame_.convert_ms = elapsedMs(t_convert);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Mesh converted to Float32MultiArray in %.3f ms (%zu vertices)",
            static_cast<unsigned long long>(frame_seq_), frame_.convert_ms, vertex_array_.data.size() / 7);
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
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<Float32ArrayNode>());
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("float_array_node"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
