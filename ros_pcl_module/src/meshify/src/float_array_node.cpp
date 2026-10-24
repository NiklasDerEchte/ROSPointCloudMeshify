#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/vtk_io.h>
#include <pcl/surface/poisson.h>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <pcl/surface/organized_fast_mesh.h>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>

class Float32ArrayNode : public rclcpp::Node {
public:
    Float32ArrayNode() 
        : Node("float_array_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          search_radius_(declare_parameter("search_radius", 0.1)),
          max_neighbors_(declare_parameter("max_neighbors", 150)),
          normal_k_search_(declare_parameter("normal_k_search", 20)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices")),
          mode_(declare_parameter("mode", "fast")), // fast, greedy, poisson
          csv_path_(declare_parameter<std::string>("csv_path", ""))
        {

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba", 100, 
            std::bind(&Float32ArrayNode::pointCloudCallback, this, std::placeholders::_1)
        );

        // Interleaved xyzrgba floats for Unity (one message, TRIANGLE_LIST order)
        mesh_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(this->output_topic_, 10);
        RCLCPP_INFO(this->get_logger(), "Node initialized, publishing Float32MultiArray on %s", output_topic_.c_str());

        // csv_path empty: no file. One data row per frame, header only on a new file.
        if (!csv_path_.empty()) {
            csv_.open(csv_path_, std::ios::out | std::ios::app);
            if (!csv_) {
                RCLCPP_WARN(this->get_logger(), "csv_path '%s' could not be opened, CSV off", csv_path_.c_str());
            } else {
                csv_.seekp(0, std::ios::end);
                if (csv_.tellp() == 0) {
                    csv_ << "seq,stamp_sec,stamp_nanosec,capture_to_callback_ms,unpack_ms,plane_ms,normals_ms,mesh_ms,convert_ms,publish_ms,total_ms,bytes,triangles\n";
                }
                csv_.setf(std::ios::fixed);
                csv_.precision(3);
                csv_.flush();
                RCLCPP_INFO(this->get_logger(), "CSV frame log: %s", csv_path_.c_str());
            }
        }
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        const auto t_frame = std::chrono::steady_clock::now();
        const unsigned long long seq = ++frame_seq_;
        stamp_valid_ = false;
        stamp_sec_ = 0;
        stamp_nanosec_ = 0;
        capture_to_callback_ms_ = 0.0;
        unpack_ms_ = 0.0;
        plane_ms_ = 0.0;
        normals_ms_ = 0.0;
        mesh_ms_ = 0.0;
        convert_ms_ = 0.0;
        publish_ms_ = 0.0;
        total_ms_ = 0.0;
        frame_bytes_ = 0;
        frame_triangles_ = 0;

        try {
            // Capture (header.stamp) to callback entry, both on the ROS clock (this->now()).
            // The subscription queue (depth 100, single-threaded spin) sits inside this
            // interval and is not split out here: callback entry is the start of processing.
            // rmw received_timestamp is stamped at take(), immediately before this callback,
            // and on the system clock, not the ROS clock, so it is not a separate queue wait.
            const rclcpp::Time received = this->now();
            stamp_sec_ = msg->header.stamp.sec;
            stamp_nanosec_ = msg->header.stamp.nanosec;
            if (stamp_sec_ != 0 || stamp_nanosec_ != 0) {
                const rclcpp::Time stamp(msg->header.stamp, received.get_clock_type());
                capture_to_callback_ms_ = (received - stamp).seconds() * 1000.0;
                stamp_valid_ = true;
                RCLCPP_INFO(this->get_logger(), "seq=%llu Capture to callback in %.3f ms (stamp %d.%09u)",
                    seq, capture_to_callback_ms_, stamp_sec_, stamp_nanosec_);
            } else {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Capture to callback n/a (header.stamp is 0)", seq);
            }

            RCLCPP_INFO(this->get_logger(), "seq=%llu Received PointCloud2 message", seq);

            bool hasRgb = false;
            bool hasRgba = false;

            for (const auto &field : msg->fields) {
                if (field.name == "rgb") {
                    hasRgb = true;
                    break;
                } else if (field.name == "rgba") {
                    hasRgba = true;
                    break;
                }
            }

            if(hasRgb || hasRgba) {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Handle XYZ-RGB(A) PointCloud", seq);
                this->handlePointCloud<pcl::PointXYZRGB, pcl::Normal, pcl::PointXYZRGBNormal>(msg);
            } else {
                RCLCPP_INFO(this->get_logger(), "seq=%llu Handle XYZ PointCloud", seq);
                this->handlePointCloud<pcl::PointXYZ, pcl::Normal, pcl::PointNormal>(msg);
            }
            total_ms_ = elapsedMs(t_frame);
            RCLCPP_INFO(this->get_logger(), "seq=%llu bytes %zu triangles %zu", seq, frame_bytes_, frame_triangles_);
            RCLCPP_INFO(this->get_logger(), "seq=%llu Total in %.3f ms", seq, total_ms_);
            writeFrameCsv();
            RCLCPP_INFO(this->get_logger(), "seq=%llu --- frame done ---", seq);

        } catch (const std::exception &e) {
            total_ms_ = elapsedMs(t_frame);
            RCLCPP_ERROR(this->get_logger(), "seq=%llu Error processing PointCloud: %s", seq, e.what());
            RCLCPP_INFO(this->get_logger(), "seq=%llu bytes %zu triangles %zu", seq, frame_bytes_, frame_triangles_);
            RCLCPP_INFO(this->get_logger(), "seq=%llu Total in %.3f ms", seq, total_ms_);
            writeFrameCsv();
        }
    }

    // ----------------------------------------------------------------------- [Handle PointCloud Message]

    template <typename PointT, typename NormalT, typename PointNormalT>
    void handlePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        const unsigned long long seq = frame_seq_;
        // Convert ROS2 PointCloud2 to PCL
        typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        const auto t_unpack = std::chrono::steady_clock::now();
        pcl::fromROSMsg(*msg, *cloud);
        unpack_ms_ = elapsedMs(t_unpack);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Unpack point cloud in %.3f ms", seq, unpack_ms_);
        pcl::PolygonMesh mesh;
        
        if (this->mode_ == "fast" && cloud->isOrganized()) {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, false, false);

            if (cloud_filtered == nullptr || cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
                return;
            }

            RCLCPP_INFO(this->get_logger(), "seq=%llu Create organized triangulation mesh", seq);
            this->createOrganizedTriangulationMesh<PointT>(cloud_filtered, mesh); 
        } else if (this->mode_ == "poisson") {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, true, true); // ToDo ist das hier richtig? weil poisson doch keine unorganized clouds bekommen darf

            if (cloud_filtered == nullptr || cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
                return;
            }
            RCLCPP_INFO(this->get_logger(), "seq=%llu Create unorganized poisson mesh", seq);
            typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals = this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

            RCLCPP_INFO(this->get_logger(), "seq=%llu Create poisson-mesh", seq);
            this->createPoissonMesh<PointNormalT>(cloud_with_normals, mesh);
        } else if (this->mode_ == "greedy" || !cloud->isOrganized()) {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, false, false);

            if (cloud_filtered == nullptr || cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "seq=%llu Filtered PointCloud is empty", seq);
                return;
            }

            RCLCPP_INFO(this->get_logger(), "seq=%llu Create unorganized greedy mesh", seq);
            typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals = this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

            RCLCPP_INFO(this->get_logger(), "seq=%llu Create greedy-triangulation-mesh", seq);
            this->createGreedyTriangulationMesh<PointNormalT>(cloud_with_normals, mesh);
        } else {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Unknown mode: %s", seq, this->mode_.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "seq=%llu Convert mesh to Float32MultiArray (xyzrgba)", seq);
        std_msgs::msg::Float32MultiArray vertex_array;
        convertMeshToFloatArray<PointT>(mesh, vertex_array);

        if (vertex_array.data.empty()) {
            RCLCPP_WARN(this->get_logger(), "seq=%llu Float32MultiArray empty, skip publish", seq);
            return;
        }

        frame_bytes_ = vertex_array.data.size() * 4;
        frame_triangles_ = (vertex_array.data.size() / 7) / 3;
        const auto t_publish = std::chrono::steady_clock::now();
        mesh_pub_->publish(vertex_array);
        publish_ms_ = elapsedMs(t_publish);
        RCLCPP_INFO(this->get_logger(), "seq=%llu publish %zu floats (%zu vertices) in %.3f ms",
            seq, vertex_array.data.size(), vertex_array.data.size() / 7, publish_ms_);
    }

    // ----------------------------------------------------------------------- [Surface Reconstruction Algos]

    template <typename PointT>
    void createOrganizedTriangulationMesh(typename pcl::PointCloud<PointT>::Ptr &cloud, pcl::PolygonMesh &mesh) {
        const auto t1 = std::chrono::steady_clock::now();
        typename pcl::OrganizedFastMesh<PointT> ofm;
        ofm.setInputCloud(cloud);
        ofm.setTrianglePixelSize(4); // Größe der Pixel für die Nachbarschaft
        ofm.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
        ofm.reconstruct(mesh);
        mesh_ms_ = elapsedMs(t1);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Organized mesh created in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), mesh_ms_);
    }

    template <typename PointNormalT>
    void createGreedyTriangulationMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
        const auto t1 = std::chrono::steady_clock::now();
        // Create mesh using greedy triangulation
        typename pcl::search::KdTree<PointNormalT>::Ptr tree(new pcl::search::KdTree<PointNormalT>());
        typename pcl::GreedyProjectionTriangulation<PointNormalT> gp3;
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
        mesh_ms_ = elapsedMs(t1);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Greedy mesh created in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), mesh_ms_);
    }

    template <typename PointNormalT>
    void createPoissonMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
        const auto t1 = std::chrono::steady_clock::now();
        // Perform Poisson Surface Reconstruction
        typename pcl::Poisson<PointNormalT> poisson;
        poisson.setDepth(8);  // Depth of reconstruction
        poisson.setSamplesPerNode(1.0f);  // Samples per node
        poisson.setSolverDivide(8);  // Solver divide
        poisson.setIsoDivide(8);  // Iso divide
        //poisson.setUsePredictedNormals(true); // Use predicted normals if available
        poisson.setInputCloud(cloud_with_normals);

        poisson.reconstruct(mesh);
        
        // ToDo Funktioniert das hier trotz Warnung?
        if constexpr (std::is_same<PointNormalT, pcl::PointXYZRGBNormal>::value) { // Workaround weil poisson reconstruction Farbinformationen löscht
            typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(*cloud_with_normals, *cloud_rgb);

            // Rekonstruiere die Punktwolke aus dem Mesh
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            pcl::fromPCLPointCloud2(mesh.cloud, *mesh_cloud);

            // Übertrage RGB-Werte auf rekonstruierte Punkte
            for (size_t i = 0; i < mesh_cloud->points.size(); ++i) {
                if (i < cloud_rgb->points.size()) {
                    mesh_cloud->points[i].r = cloud_rgb->points[i].r;
                    mesh_cloud->points[i].g = cloud_rgb->points[i].g;
                    mesh_cloud->points[i].b = cloud_rgb->points[i].b;
                }
            }

            // Konvertiere zurück in PCLPointCloud2 und aktualisiere mesh.cloud
            pcl::toPCLPointCloud2(*mesh_cloud, mesh.cloud);
        }
        mesh_ms_ = elapsedMs(t1);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Poisson mesh created in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), mesh_ms_);
    }

    // ----------------------------------------------------------------------- [Convert Mesh -> Float32MultiArray xyzrgba]

    template <typename PointT>
    void convertMeshToFloatArray(
        const pcl::PolygonMesh &mesh,
        std_msgs::msg::Float32MultiArray &vertex_array) {
        const auto t1 = std::chrono::steady_clock::now();
        const unsigned long long seq = frame_seq_;

        typename pcl::PointCloud<PointT> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        if (mesh.polygons.empty()) {
            convert_ms_ = elapsedMs(t1);
            RCLCPP_WARN(this->get_logger(), "seq=%llu Triangulation failed: no polygons created", seq);
            return;
        }

        size_t tri = 0;
        for (const auto &p : mesh.polygons) {
            if (p.vertices.size() == 3) {
                ++tri;
            }
        }
        
        vertex_array.data.resize(tri * 3 * 7);
        float *d = vertex_array.data.data();
        size_t w = 0;
        for (const auto &polygon : mesh.polygons) {
            if (polygon.vertices.size() != 3) {
                continue;
            }

            for (const auto vertex_idx : polygon.vertices) {
                const auto &point = cloud.points[vertex_idx];

                d[w + 0] = point.x;
                d[w + 1] = point.y;
                d[w + 2] = point.z;

                if constexpr (std::is_same_v<PointT, pcl::PointXYZRGB> ||
                            std::is_same_v<PointT, pcl::PointXYZRGBA>) {
                    d[w + 3] = point.r / 255.0f;
                    d[w + 4] = point.g / 255.0f;
                    d[w + 5] = point.b / 255.0f;
                    d[w + 6] = 1.0f;
                } else {
                    d[w + 3] = 0.0f;
                    d[w + 4] = 1.0f;
                    d[w + 5] = 0.0f;
                    d[w + 6] = 1.0f;
                }
                w += 7;
            }
        }

        convert_ms_ = elapsedMs(t1);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Mesh converted to Float32MultiArray in %.3f ms (%zu vertices)",
            seq, convert_ms_, vertex_array.data.size() / 7);
    }

    // ----------------------------------------------------------------------- [Helper functions]

    template <typename PointT, typename NormalT, typename PointNormalT>
    typename pcl::PointCloud<PointNormalT>::Ptr estimateNormals(typename pcl::PointCloud<PointT>::Ptr &cloud) {
        const auto t1 = std::chrono::steady_clock::now();
        // Estimate normals
        typename pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>());
        typename pcl::search::KdTree<PointT>::Ptr tree_xyz(new pcl::search::KdTree<PointT>());
        typename pcl::NormalEstimation<PointT, NormalT> ne;
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree_xyz);
        ne.setKSearch(normal_k_search_);
        ne.compute(*normals);

        typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals(new pcl::PointCloud<PointNormalT>());
        pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
        normals_ms_ = elapsedMs(t1);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Normals estimated in %.3f ms",
            static_cast<unsigned long long>(frame_seq_), normals_ms_);
        return cloud_with_normals;
    }

    /**
     * removePlanePoints und removeNaNValues können aus einer organisierten pointcloud eine unorganisierte machen
    */
    template <typename PointT>
    typename pcl::PointCloud<PointT>::Ptr planeSegmentation(
        typename pcl::PointCloud<PointT>::Ptr &cloudIn,
        bool removePlanePoints, 
        bool removeNaNValues) {
        const auto t1 = std::chrono::steady_clock::now();
        const unsigned long long seq = frame_seq_;
        typename pcl::PointCloud<PointT>::Ptr cloudOut(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud<PointT, PointT>(*cloudIn, *cloudOut);

        // RANSAC only samples finite pixels. The cloud stays organized
        pcl::IndicesPtr finite(new pcl::Indices());
        finite->reserve(cloudOut->size());
        for (std::size_t i = 0; i < cloudOut->points.size(); ++i) {
            const auto &p = cloudOut->points[i];
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                finite->push_back(static_cast<int>(i));
            }
        }

        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane RANSAC on %zu/%zu finite points (%u x %u)",
            seq, finite->size(), cloudOut->size(), cloudOut->width, cloudOut->height);

        if (finite->size() < 3) {
            plane_ms_ = elapsedMs(t1);
            RCLCPP_WARN(this->get_logger(), "seq=%llu Not enough finite points for plane segmentation", seq);
            return nullptr;
        }

        // Perform plane segmentation
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<PointT> seg;

        seg.setOptimizeCoefficients(true);
        seg.setMaxIterations(1000);
        seg.setProbability(0.85);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold_);
        seg.setInputCloud(cloudOut);
        seg.setIndices(finite);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            plane_ms_ = elapsedMs(t1);
            RCLCPP_WARN(this->get_logger(), "seq=%llu No plane found in the point cloud", seq);
            return nullptr;
        }

        if(removeNaNValues) {
            // Remove NaN values
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*cloudOut, *cloudOut, indices);
        }

        if(removePlanePoints) {
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(cloudOut);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloudOut);
        } else {
            for (int idx : inliers->indices) {
                cloudOut->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                cloudOut->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                cloudOut->points[idx].z = std::numeric_limits<float>::quiet_NaN();
            }
        }
        plane_ms_ = elapsedMs(t1);
        RCLCPP_INFO(this->get_logger(), "seq=%llu Plane segmentation in %.3f ms", seq, plane_ms_);
        return cloudOut;
    }

    // Stage durations are steady_clock milliseconds. Capture-to-callback is ROS time, not this.
    static double elapsedMs(std::chrono::steady_clock::time_point t0)
    {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    // capture_to_callback_ms is left empty when header.stamp is zero.
    void writeFrameCsv()
    {
        if (!csv_.is_open()) {
            return;
        }
        csv_ << frame_seq_ << ','
             << stamp_sec_ << ','
             << stamp_nanosec_ << ',';
        if (stamp_valid_) {
            csv_ << capture_to_callback_ms_;
        }
        csv_ << ','
             << unpack_ms_ << ','
             << plane_ms_ << ','
             << normals_ms_ << ','
             << mesh_ms_ << ','
             << convert_ms_ << ','
             << publish_ms_ << ','
             << total_ms_ << ','
             << frame_bytes_ << ','
             << frame_triangles_ << '\n';
        csv_.flush();
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mesh_pub_;

    double distance_threshold_;
    double search_radius_;
    int max_neighbors_;
    int normal_k_search_;
    std::string output_topic_;
    std::string mode_;
    std::string csv_path_;

    // rclcpp::spin is single-threaded, so one frame owns these fields.
    uint64_t frame_seq_{0};
    bool stamp_valid_{false};
    int32_t stamp_sec_{0};
    uint32_t stamp_nanosec_{0};
    double capture_to_callback_ms_{0.0};
    double unpack_ms_{0.0};
    double plane_ms_{0.0};
    double normals_ms_{0.0};
    double mesh_ms_{0.0};
    double convert_ms_{0.0};
    double publish_ms_{0.0};
    double total_ms_{0.0};
    size_t frame_bytes_{0};
    size_t frame_triangles_{0};
    std::ofstream csv_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Float32ArrayNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
