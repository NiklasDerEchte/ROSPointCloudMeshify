#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>
#include <visualization_msgs/msg/marker_array.hpp>

class ObjectDetectionMeshCreatorNode : public rclcpp::Node {
public:
    ObjectDetectionMeshCreatorNode() : Node("object_detection_mesh_creator_node") {
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points/xyzrgba", 10,
            std::bind(&ObjectDetectionMeshCreatorNode::pointCloudCallback, this, std::placeholders::_1));
        mesh_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/object_markers", 10);

        // Konfigurierbare Parameter
        this->declare_parameter("plane_distance_threshold", 0.01);
        this->declare_parameter("cluster_tolerance", 0.02);
        this->declare_parameter("min_cluster_size", 100);
        this->declare_parameter("max_cluster_size", 25000);
        this->declare_parameter("normal_radius", 0.03);
        this->declare_parameter("mesh_search_radius", 0.1);
        RCLCPP_INFO(this->get_logger(), "%s%s", this->get_name(), " - READY");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Messange empfangen");
        // Konvertiere ROS-PointCloud2 in PCL-PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Eingehende Punktwolke ist leer.");
            return;
        }

        // NaN-Werte entfernen
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

        if (cloud_filtered->empty()) {
            RCLCPP_WARN(this->get_logger(), "Nach Entfernung der NaN-Werte ist die Punktwolke leer.");
            return;
        }

        // Ebene entfernen
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<pcl::PointXYZ> seg;

        double plane_distance_threshold;
        this->get_parameter("plane_distance_threshold", plane_distance_threshold);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(plane_distance_threshold);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "Keine Ebene gefunden.");
        } else {
            pcl::ExtractIndices<pcl::PointXYZ> extract;
            extract.setInputCloud(cloud_filtered);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloud_filtered);
        }

        if (cloud_filtered->empty()) {
            RCLCPP_WARN(this->get_logger(), "Nach Entfernung der Ebene ist die Punktwolke leer.");
            return;
        }

        // Cluster-Extraktion
        pcl::search::KdTree<pcl::PointXYZ>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZ>());
        cluster_tree->setInputCloud(cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

        double cluster_tolerance;
        int min_cluster_size, max_cluster_size;
        this->get_parameter("cluster_tolerance", cluster_tolerance);
        this->get_parameter("min_cluster_size", min_cluster_size);
        this->get_parameter("max_cluster_size", max_cluster_size);

        ec.setClusterTolerance(cluster_tolerance);
        ec.setMinClusterSize(min_cluster_size);
        ec.setMaxClusterSize(max_cluster_size);
        ec.setSearchMethod(cluster_tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        if (cluster_indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "Keine Cluster gefunden.");
            return;
        }

        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;

        for (const auto &indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto &idx : indices.indices) {
                object_cloud->points.push_back(cloud_filtered->points[idx]);
            }

            // Bounding Box prüfen
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*object_cloud, min_pt, max_pt);
            float volume = (max_pt.x - min_pt.x) * (max_pt.y - min_pt.y) * (max_pt.z - min_pt.z);
            if (volume < 0.001) {
                RCLCPP_INFO(this->get_logger(), "Cluster mit zu kleinem Volumen ignoriert.");
                continue;
            }

            // Normalenschätzung
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz(new pcl::search::KdTree<pcl::PointXYZ>());
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;

            double normal_radius;
            this->get_parameter("normal_radius", normal_radius);

            ne.setInputCloud(object_cloud);
            ne.setSearchMethod(tree_xyz);
            ne.setRadiusSearch(normal_radius);
            ne.compute(*normals);

            pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
            pcl::concatenateFields(*object_cloud, *normals, *cloud_with_normals);

            // Mesherstellung mit Greedy Triangulation
            pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
            pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
            pcl::PolygonMesh mesh;

            double mesh_search_radius;
            this->get_parameter("mesh_search_radius", mesh_search_radius);

            gp3.setSearchRadius(mesh_search_radius);
            gp3.setMu(2.5);
            gp3.setMaximumNearestNeighbors(150);
            gp3.setMaximumSurfaceAngle(M_PI / 4);
            gp3.setMinimumAngle(M_PI / 18);
            gp3.setMaximumAngle(2 * M_PI / 3);
            gp3.setNormalConsistency(false);

            gp3.setInputCloud(cloud_with_normals);
            gp3.setSearchMethod(tree);
            gp3.reconstruct(mesh);

            convertMeshToMarkers(mesh, marker_array, marker_id);
            marker_id++;
        }

        // Marker veröffentlichen
        mesh_pub_->publish(marker_array);
    }

    void convertMeshToMarkers(const pcl::PolygonMesh &mesh, visualization_msgs::msg::MarkerArray &marker_array, int marker_id) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "zivid_optical_frame";
        marker.header.stamp = this->now();
        marker.ns = "mesh";
        marker.id = marker_id;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        pcl::PointCloud<pcl::PointXYZ> cloud;
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
                }
            }
        }

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;

        marker_array.markers.push_back(marker);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr mesh_pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjectDetectionMeshCreatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
