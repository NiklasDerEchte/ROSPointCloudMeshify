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

class ObjectDetectionMeshCreatorNode : public rclcpp::Node {
public:
    ObjectDetectionMeshCreatorNode() 
        : Node("object_detection_mesh_creator_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          search_radius_(declare_parameter("search_radius", 0.1)),
          max_neighbors_(declare_parameter("max_neighbors", 150)),
          normal_k_search_(declare_parameter("normal_k_search", 20)) {

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points/xyzrgba", 10, 
            std::bind(&ObjectDetectionMeshCreatorNode::pointCloudCallback, this, std::placeholders::_1)
        );

        mesh_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/object_markers", 10);
        RCLCPP_INFO(this->get_logger(), "Node initialized and ready to process PointCloud2 messages.");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        try {
            RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message.");

            bool hasRgb = false;
            bool hasRgba = false;
            bool isOrganized = false;

            for (const auto &field : msg->fields) {
                if (field.name == "rgb") {
                    hasRgb = true;
                } else if (field.name == "rgba") {
                    hasRgba = true;
                }
            }

            // Convert ROS2 PointCloud2 to PCL
            if(hasRgb) {
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
                pcl::fromROSMsg(*msg, *cloud);
                isOrganized = cloud->isOrganized();
                RCLCPP_INFO(this->get_logger(), ("Handle RGB PointCloud2 message. [Organized=" + std::to_string(isOrganized) + "]").c_str());
                this->handle<pcl::PointXYZRGB>(hasRgb, hasRgba, isOrganized, *cloud);
            } else if (hasRgba) {
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());
                pcl::fromROSMsg(*msg, *cloud);
                isOrganized = cloud->isOrganized();
                RCLCPP_INFO(this->get_logger(), ("Handle RGBA PointCloud2 message. [Organized=" + std::to_string(isOrganized) + "]").c_str());
                this->handle<pcl::PointXYZRGBA>(hasRgb, hasRgba, isOrganized, *cloud);
            } else {
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::fromROSMsg(*msg, *cloud);
                isOrganized = cloud->isOrganized();
                RCLCPP_INFO(this->get_logger(), ("Handle Plain PointCloud2 message. [Organized=" + std::to_string(isOrganized) + "]").c_str());
                this->handle<pcl::PointXYZ>(hasRgb, hasRgba, isOrganized, *cloud);
            }

        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing PointCloud: %s", e.what());
        }
    }

    template <typename T>
    void handle(bool hasRgb, bool hasRgba, bool isOrganized, pcl::PointCloud<T> &cloud) {
        // Remove NaN values
        pcl::PointCloud<T>::Ptr cloud_filtered(new pcl::PointCloud<T>());
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

        // Perform plane segmentation and remove the plane
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<T> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(distance_threshold_);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "No plane found in the point cloud.");
            return;
        }

        pcl::ExtractIndices<T> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_filtered);

        pcl::PolygonMesh mesh;
        if (isOrganized) {
            this->createOrganizedTriangulationMesh<T>(cloud_filtered, mesh);
        } else {
            // Estimate normals
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            pcl::search::KdTree<T>::Ptr tree_xyz(new pcl::search::KdTree<T>());
            pcl::NormalEstimation<T, pcl::Normal> ne;
            ne.setInputCloud(cloud_filtered);
            ne.setSearchMethod(tree_xyz);
            ne.setKSearch(normal_k_search_);
            ne.compute(*normals);

            pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
            pcl::concatenateFields(*cloud_filtered, *normals, *cloud_with_normals);
            this->createGreedyTriangulationMesh(cloud_with_normals, mesh); // oder poisson
        }

        // Convert mesh to markers and publish
        visualization_msgs::msg::MarkerArray marker_array;
        convertMeshToMarkers(hasRgb, hasRgba, isOrganized, mesh, marker_array);
        mesh_pub_->publish(marker_array);

    }

    template <typename T>
    void createOrganizedTriangulationMesh(pcl::PointCloud<T> &cloud, pcl::PolygonMesh &mesh) {
        pcl::OrganizedFastMesh<T> ofm;
        ofm.setInputCloud(cloud);
        ofm.setTrianglePixelSize(1); // Größe der Pixel für die Nachbarschaft
        ofm.setTriangulationType(pcl::OrganizedFastMesh<T>::TRIANGLE_ADAPTIVE_CUT);
        ofm.reconstruct(mesh);
    }

    void createGreedyTriangulationMesh(pcl::PointCloud<pcl::PointNormal>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
        // Create mesh using greedy triangulation
        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
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
    }

    void createPoissonMesh(const pcl::PointCloud<pcl::PointNormal>::ConstPtr &cloud_with_normals, pcl::PolygonMesh &mesh) {
        // Perform Poisson Surface Reconstruction
        pcl::Poisson<pcl::PointNormal> poisson;
        poisson.setDepth(8);  // Depth of reconstruction
        poisson.setSamplesPerNode(1.0f);  // Samples per node
        poisson.setSolverDivide(8);  // Solver divide
        poisson.setIsoDivide(8);  // Iso divide
        //poisson.setUsePredictedNormals(true); // Use predicted normals if available
        poisson.setInputCloud(cloud_with_normals);

        poisson.reconstruct(mesh);
    }

    template <typename T>
    void convertMeshToMarkers(bool hasRgb, bool hasRgba, bool isOrganized, const pcl::PolygonMesh &mesh, visualization_msgs::msg::MarkerArray &marker_array) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "zivid_optical_frame";
        marker.header.stamp = this->now();
        marker.ns = "mesh";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        pcl::PointCloud<T> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        for (const auto &polygon : mesh.polygons) {
            if (polygon.vertices.size() == 3) {
                for (const auto &vertex_idx : polygon.vertices) {
                    const auto &point = cloud.points[vertex_idx];
                    geometry_msgs::msg::Point pt;
                    pt.x = point.x;
                    pt.y = point.y;
                    pt.z = point.z;
                    marker.points.push_back(pt);

                    if(hasRgb || hasRgba) {
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

        if(!hasRgb && !hasRgba) {
            marker.color.a = 1.0;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;
        }

        marker_array.markers.push_back(marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mesh_pub_;

    double distance_threshold_;
    double search_radius_;
    int max_neighbors_;
    int normal_k_search_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetectionMeshCreatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
