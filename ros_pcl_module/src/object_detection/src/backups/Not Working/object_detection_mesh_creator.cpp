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
            "/points/xyzrgba", 
            10,
            std::bind(
                &ObjectDetectionMeshCreatorNode::pointCloudCallback, 
                this, 
                std::placeholders::_1
            )
        );
        mesh_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/object_markers", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // 1. NaN-Werte entfernen
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<int> indices; // Dummy für die Indizes
        pcl::removeNaNFromPointCloud(*cloud, *cloud_filtered, indices);

        // 2. Ebene entfernen (Plane Segmentation)
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud_filtered);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "Keine Ebene gefunden.");
            return;
        }

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*cloud_filtered);

        // 3. Cluster-Extraktion
        pcl::search::KdTree<pcl::PointXYZ>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZ>());
        cluster_tree->setInputCloud(cloud_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02);
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(cluster_tree);
        ec.setInputCloud(cloud_filtered);
        ec.extract(cluster_indices);

        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;

        for (const auto &indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto &idx : indices.indices) {
                object_cloud->points.push_back(cloud_filtered->points[idx]);
            }

            // Bounding Box überprüfen (optional, falls erforderlich)
            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*object_cloud, min_pt, max_pt);
            float volume = (max_pt.x - min_pt.x) * (max_pt.y - min_pt.y) * (max_pt.z - min_pt.z);
            if (volume < 0.001) {
                continue;
            }

            // 4. Normalenschätzung
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz(new pcl::search::KdTree<pcl::PointXYZ>());
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
            ne.setInputCloud(object_cloud);
            ne.setSearchMethod(tree_xyz);
            ne.setRadiusSearch(0.03);
            ne.compute(*normals);

            pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
            pcl::concatenateFields(*object_cloud, *normals, *cloud_with_normals);

            // 5. Mesherstellung mit Greedy Triangulation
            pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
            pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
            pcl::PolygonMesh mesh;
            gp3.setSearchRadius(0.1);
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

        // 6. Veröffentlichen der Marker
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
