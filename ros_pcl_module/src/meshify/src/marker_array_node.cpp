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
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/surface/organized_fast_mesh.h>

class MarkerArrayNode : public rclcpp::Node {
public:
    MarkerArrayNode() 
        : Node("marker_array_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          search_radius_(declare_parameter("search_radius", 0.1)),
          max_neighbors_(declare_parameter("max_neighbors", 150)),
          normal_k_search_(declare_parameter("normal_k_search", 20)),
          output_topic_(declare_parameter("output_topic", "/object_markers")),
          mode_(declare_parameter("mode", "fast")) // fast, greedy, poisson
        {

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba", 10, 
            std::bind(&MarkerArrayNode::pointCloudCallback, this, std::placeholders::_1)
        );

        mesh_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(this->output_topic_, 10);
        RCLCPP_INFO(this->get_logger(), "Node initialized");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        try {
            RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");

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
                RCLCPP_INFO(this->get_logger(), "Handle XYZ-RGB(A) PointCloud");
                this->handlePointCloud<pcl::PointXYZRGB, pcl::Normal, pcl::PointXYZRGBNormal>(msg);
            } else {
                RCLCPP_INFO(this->get_logger(), "Handle XYZ PointCloud");
                this->handlePointCloud<pcl::PointXYZ, pcl::Normal, pcl::PointNormal>(msg);
            }

        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing PointCloud: %s", e.what());
        }
    }

    // ----------------------------------------------------------------------- [Handle PointCloud Message]

    template <typename PointT, typename NormalT, typename PointNormalT>
    void handlePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert ROS2 PointCloud2 to PCL
        typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*msg, *cloud);
        pcl::PolygonMesh mesh;
        
        if (this->mode_ == "fast" && cloud->isOrganized()) {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, false, false);

            if (cloud_filtered == nullptr || cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Create organized triangulation mesh");
            this->createOrganizedTriangulationMesh<PointT>(cloud_filtered, mesh); 
        } else if (this->mode_ == "poisson") {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, true, true); // ToDo ist das hier richtig? weil poisson doch keine unorganized clouds bekommen darf

            if (cloud_filtered == nullptr || cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Create unorganized poisson mesh");
            typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals = this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

            RCLCPP_INFO(this->get_logger(), "Create poisson-mesh");
            this->createPoissonMesh<PointNormalT>(cloud_with_normals, mesh);
        } else if (this->mode_ == "greedy" || !cloud->isOrganized()) {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, false, false);

            if (cloud_filtered == nullptr || cloud_filtered->empty()) {
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Create unorganized greedy mesh");
            typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals = this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

            RCLCPP_INFO(this->get_logger(), "Create greedy-triangulation-mesh");
            this->createGreedyTriangulationMesh<PointNormalT>(cloud_with_normals, mesh);
        } else {
            RCLCPP_WARN(this->get_logger(), "Unknown mode: %s", this->mode_.c_str());
            return;
        }

        for (const auto &field : mesh.cloud.fields) {
            std::cout << "Field name: " << field.name << std::endl;
        }


        RCLCPP_INFO(this->get_logger(), "Convert mesh to markers");
        // Convert mesh to markers and publish
        visualization_msgs::msg::MarkerArray marker_array;
        convertMeshToMarkers<PointT>(mesh, marker_array);

        RCLCPP_INFO(this->get_logger(), "publish");
        mesh_pub_->publish(marker_array);
    }

    // ----------------------------------------------------------------------- [Surface Reconstruction Algos]

    template <typename PointT>
    void createOrganizedTriangulationMesh(typename pcl::PointCloud<PointT>::Ptr &cloud, pcl::PolygonMesh &mesh) {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        typename pcl::OrganizedFastMesh<PointT> ofm;
        ofm.setInputCloud(cloud);
        ofm.setTrianglePixelSize(4); // Größe der Pixel für die Nachbarschaft
        ofm.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
        ofm.reconstruct(mesh);
        RCLCPP_INFO(this->get_logger(), "Organized mesh created in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
    }

    template <typename PointNormalT>
    void createGreedyTriangulationMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
        RCLCPP_INFO(this->get_logger(), "Greedy mesh created in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
    }

    template <typename PointNormalT>
    void createPoissonMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
        RCLCPP_INFO(this->get_logger(), "Poisson mesh created in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
    }

    // ----------------------------------------------------------------------- [Convert Mesh -> Markers]

    template <typename PointT>
    void convertMeshToMarkers(
        const pcl::PolygonMesh &mesh, 
        visualization_msgs::msg::MarkerArray &marker_array) {
        visualization_msgs::msg::Marker marker;
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        marker.header.frame_id = "zivid_optical_frame";
        marker.header.stamp = this->now();
        marker.ns = "mesh";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        typename pcl::PointCloud<PointT> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        if (mesh.polygons.empty()) {
            RCLCPP_WARN(this->get_logger(), "Triangulation failed: no polygons created");
            return;
        }

        for (const auto &polygon : mesh.polygons) {
            if (polygon.vertices.size() == 3) {
                for (const auto &vertex_idx : polygon.vertices) {
                    const auto &point = cloud.points[vertex_idx];
                    geometry_msgs::msg::Point pt;
                    pt.x = point.x;
                    pt.y = point.y;
                    pt.z = point.z;
                    marker.points.push_back(pt);

                    if constexpr (std::is_same<PointT, pcl::PointXYZRGB>::value || std::is_same<PointT, pcl::PointXYZRGBA>::value) { 
                        std_msgs::msg::ColorRGBA color;
                        color.a = 1.0;
                        color.r = static_cast<float>(point.r) / 255.0f;
                        color.g = static_cast<float>(point.g) / 255.0f;
                        color.b = static_cast<float>(point.b) / 255.0f;
                        marker.colors.push_back(color);
                    }
                }
            }
        }

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;

        if constexpr (std::is_same<PointT, pcl::PointXYZ>::value) {
            marker.color.a = 1.0;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }
        
        marker_array.markers.push_back(marker);
        RCLCPP_INFO(this->get_logger(), "Mesh converted to markers in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
    }

    // ----------------------------------------------------------------------- [Helper functions]

    template <typename PointT, typename NormalT, typename PointNormalT>
    typename pcl::PointCloud<PointNormalT>::Ptr estimateNormals(typename pcl::PointCloud<PointT>::Ptr &cloud) {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
        RCLCPP_INFO(this->get_logger(), "Normals estimated in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
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
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        typename pcl::PointCloud<PointT>::Ptr cloudOut(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud<PointT, PointT>(*cloudIn, *cloudOut);

        // Perform plane segmentation
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<PointT> seg;

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold_);
        seg.setInputCloud(cloudOut);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No plane found in the point cloud");
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
        RCLCPP_INFO(this->get_logger(), "Plane segmentation in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
        return cloudOut;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mesh_pub_;

    double distance_threshold_;
    double search_radius_;
    int max_neighbors_;
    int normal_k_search_;
    std::string output_topic_;
    std::string mode_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MarkerArrayNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}