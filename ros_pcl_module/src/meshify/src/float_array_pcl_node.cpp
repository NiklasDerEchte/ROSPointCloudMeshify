#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/poisson.h>

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
 * PCL mesh node for unorganized surface reconstruction. Subscribes to a PointCloud2, removes
 * the dominant plane with RANSAC, estimates normals, meshes the rest and publishes the
 * triangles as a Float32MultiArray (TRIANGLE_LIST, 7 floats per vertex: x, y, z, r, g, b, a).
 *
 * mode: greedy (GreedyProjectionTriangulation) or poisson. The organized fast mesh is
 * float_array_node (CPU) and float_array_cuda_node (GPU).
 */
class Float32ArrayPclNode : public rclcpp::Node {
public:
    /// Declare parameters, open the CSV log and subscribe to /points/xyzrgba.
    Float32ArrayPclNode()
        : Node("float_array_pcl_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices")),
          mode_(declare_parameter("mode", "greedy")),
          enable_logging_(declare_parameter("enable_logging", true)),
          csv_path_(declare_parameter<std::string>("csv_path", "meshify_pcl.csv")),
          queue_depth_(declare_parameter("queue_depth", 100)),
          search_radius_(declare_parameter("search_radius", 0.1)),
          max_neighbors_(declare_parameter("max_neighbors", 150)),
          normal_k_search_(declare_parameter("normal_k_search", 20))
    {
        if (mode_ != "greedy" && mode_ != "poisson") {
            throw std::invalid_argument("mode must be greedy or poisson, got '" + mode_ + "'");
        }

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba",
            static_cast<std::size_t>(std::max<int64_t>(1, queue_depth_)),
            std::bind(&Float32ArrayPclNode::pointCloudCallback, this, std::placeholders::_1));
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
    static constexpr const char * kBackend = "pcl";

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
                handlePointCloud<pcl::PointXYZRGB, pcl::PointXYZRGBNormal>(msg);
            } else {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Handle XYZ PointCloud", seq);
                handlePointCloud<pcl::PointXYZ, pcl::PointNormal>(msg);
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

    template <typename PointT, typename PointNormalT>
    void handlePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr & msg) {
        const unsigned long long seq = frame_seq_;

        const auto t_unpack = std::chrono::steady_clock::now();
        typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*msg, *cloud);
        frame_.unpack_ms = elapsedMs(t_unpack);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Unpack point cloud in %.3f ms", seq, frame_.unpack_ms);

        // Poisson needs a dense cloud: drop plane points and NaNs. Greedy keeps the layout and
        // only sets the plane points to NaN.
        const bool remove_points = mode_ == "poisson";
        typename pcl::PointCloud<PointT>::Ptr cloud_filtered = planeSegmentation<PointT>(cloud, remove_points);
        if (cloud_filtered == nullptr) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
            return;
        }
        if (cloud_filtered->empty()) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
            frame_.status = "empty_cloud";
            return;
        }

        typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals =
            estimateNormals<PointT, PointNormalT>(cloud_filtered);
        pcl::PolygonMesh mesh;
        if (mode_ == "poisson") {
            RCLCPP_INFO(this->get_logger(), "seq=%llu Create Poisson mesh", seq);
            createPoissonMesh<PointNormalT>(cloud_with_normals, mesh);
        } else {
            RCLCPP_INFO(this->get_logger(), "seq=%llu Create greedy triangulation mesh", seq);
            createGreedyTriangulationMesh<PointNormalT>(cloud_with_normals, mesh);
        }

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

    // RANSAC plane on the finite points. With remove_points=false the plane inliers are set
    // to NaN and the cloud stays organized; with true, inliers and NaNs are removed.
    template <typename PointT>
    typename pcl::PointCloud<PointT>::Ptr planeSegmentation(
        const typename pcl::PointCloud<PointT>::Ptr & cloud_in, bool remove_points) {
        const unsigned long long seq = frame_seq_;

        // Working copy, its own stage: the GPU segments its device buffer in place.
        const auto t_copy = std::chrono::steady_clock::now();
        typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>(*cloud_in));
        frame_.copy_ms = elapsedMs(t_copy);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Copy point cloud in %.3f ms", seq, frame_.copy_ms);

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
            logRansacInput(finite->size(), cloud->size(), cloud->width, cloud->height);
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
            logRansacInput(finite->size(), cloud->size(), cloud->width, cloud->height);
            RCLCPP_WARN(this->get_logger(), "seq=%llu No plane found in the point cloud", seq);
            frame_.status = "no_plane";
            return nullptr;
        }

        if (remove_points) {
            // Inlier indices refer to the full cloud, so extract them before dropping NaNs.
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud);
            pcl::Indices kept;
            pcl::removeNaNFromPointCloud(*cloud, *cloud, kept);
        } else {
            constexpr float nan = std::numeric_limits<float>::quiet_NaN();
            for (const int idx : inliers->indices) {
                cloud->points[idx].x = nan;
                cloud->points[idx].y = nan;
                cloud->points[idx].z = nan;
            }
        }
        frame_.plane_ms = elapsedMs(t_plane);
        logRansacInput(finite->size(), cloud_in->size(), cloud_in->width, cloud_in->height);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane segmentation in %.3f ms", seq, frame_.plane_ms);
        return cloud;
    }

    void logRansacInput(std::size_t nfinite, std::size_t npoints, unsigned width, unsigned height) {
        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane RANSAC on %zu/%zu finite points (%u x %u)",
            static_cast<unsigned long long>(frame_seq_), nfinite, npoints, width, height);
    }

    // ----------------------------------------------------------------------- [Normals]

    template <typename PointT, typename PointNormalT>
    typename pcl::PointCloud<PointNormalT>::Ptr estimateNormals(const typename pcl::PointCloud<PointT>::Ptr & cloud) {
        const auto t_normals = std::chrono::steady_clock::now();
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setKSearch(normal_k_search_);
        ne.compute(*normals);

        typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals(new pcl::PointCloud<PointNormalT>());
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
        frame_.normals_ms = elapsedMs(t_normals);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Normals estimated in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), frame_.normals_ms);
        return cloud_with_normals;
    }

    // ----------------------------------------------------------------------- [Mesh]

    template <typename PointNormalT>
    void createGreedyTriangulationMesh(
        const typename pcl::PointCloud<PointNormalT>::Ptr & cloud_with_normals, pcl::PolygonMesh & mesh) {
        const auto t_mesh = std::chrono::steady_clock::now();
        typename pcl::search::KdTree<PointNormalT>::Ptr tree(new pcl::search::KdTree<PointNormalT>());
        pcl::GreedyProjectionTriangulation<PointNormalT> gp3;
        gp3.setSearchRadius(search_radius_);
        gp3.setMu(4);
        gp3.setMaximumNearestNeighbors(max_neighbors_);
        gp3.setMaximumSurfaceAngle(M_PI / 4);
        gp3.setMinimumAngle(M_PI / 36);
        gp3.setMaximumAngle(M_PI / 2);
        gp3.setNormalConsistency(false);
        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree);
        gp3.reconstruct(mesh);
        frame_.mesh_ms = elapsedMs(t_mesh);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Greedy mesh created in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), frame_.mesh_ms);
    }

    template <typename PointNormalT>
    void createPoissonMesh(const typename pcl::PointCloud<PointNormalT>::Ptr & cloud_with_normals, pcl::PolygonMesh & mesh) {
        const auto t_mesh = std::chrono::steady_clock::now();
        pcl::Poisson<PointNormalT> poisson;
        poisson.setDepth(8);
        poisson.setSamplesPerNode(1.0f);
        poisson.setSolverDivide(8);
        poisson.setIsoDivide(8);
        poisson.setInputCloud(cloud_with_normals);
        poisson.reconstruct(mesh);

        // Poisson drops the colors. Copy them back by index; the reconstructed vertices do not
        // correspond to the input points, so this is only a rough coloring.
        if constexpr (std::is_same_v<PointNormalT, pcl::PointXYZRGBNormal>) {
            pcl::PointCloud<pcl::PointXYZRGBNormal> mesh_cloud;
            pcl::fromPCLPointCloud2(mesh.cloud, mesh_cloud);
            const std::size_t n = std::min(mesh_cloud.size(), cloud_with_normals->size());
            for (std::size_t i = 0; i < n; ++i) {
                mesh_cloud.points[i].r = cloud_with_normals->points[i].r;
                mesh_cloud.points[i].g = cloud_with_normals->points[i].g;
                mesh_cloud.points[i].b = cloud_with_normals->points[i].b;
            }
            pcl::toPCLPointCloud2(mesh_cloud, mesh.cloud);
        }
        frame_.mesh_ms = elapsedMs(t_mesh);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Poisson mesh created in %.3f ms",
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
    double search_radius_;
    int max_neighbors_;
    int normal_k_search_;

    // rclcpp::spin is single-threaded, so one frame owns these fields.
    uint64_t frame_seq_{0};
    meshify::FrameTimes frame_;
    std::ofstream csv_;
    std_msgs::msg::Float32MultiArray vertex_array_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    try {
        rclcpp::spin(std::make_shared<Float32ArrayPclNode>());
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("float_array_pcl_node"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
