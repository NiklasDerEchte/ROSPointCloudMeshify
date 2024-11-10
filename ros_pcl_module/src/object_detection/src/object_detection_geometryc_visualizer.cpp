#include <memory>
#include <vector>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl/segmentation/extract_clusters.h"
#include "pcl/filters/extract_indices.h"        // Für ExtractIndices
#include "pcl/common/common.h"
#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"                   // Für Eigen-Berechnungen
#include "pcl_conversions/pcl_conversions.h"    // Für die Konvertierung zwischen ROS und PCL
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <Eigen/Dense>                          // Eigen-Bibliothek für SelfAdjointEigenSolver
#include <Eigen/Eigenvalues>                    // Für SelfAdjointEigenSolver

class ObjectDetectionVisualizerNode : public rclcpp::Node {
public:
    ObjectDetectionVisualizerNode() : Node("object_detection_geometrcy_visualizer_node")
    {
        // Publisher für Marker
        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("object_markers", 10);

        // Subscriber für PointCloud2
        pointcloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "points/xyzrgba", 10, std::bind(&ObjectDetectionVisualizerNode::pointcloudCallback, this, std::placeholders::_1)
        );
    }
private:
    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Eingabepunktwolke ist leer. Überspringe Verarbeitung.");
            return;
        }

        // Voxelgrid-Filter anwenden
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(cloud);
        voxel_filter.setLeafSize(0.01f, 0.01f, 0.01f);
        voxel_filter.filter(*filtered_cloud);

        // Ebene segmentieren
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(filtered_cloud);
        seg.segment(*inliers, *coefficients);

        // Punktwolke ohne die Ebene extrahieren
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(filtered_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        extract.filter(*objects_cloud);

        // KdTree für Cluster-Extraktion
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(objects_cloud);

        // Euclidean Cluster Extraction
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.02);
        ec.setMinClusterSize(100);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(objects_cloud);
        ec.extract(cluster_indices);

        visualization_msgs::msg::MarkerArray marker_array;
        int id = 0;

        for (const auto& indices : cluster_indices) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : indices.indices) {
                cluster_cloud->points.push_back(objects_cloud->points[idx]);
            }

            if (cluster_cloud->points.size() < 3) {
                RCLCPP_WARN(this->get_logger(), "Cluster enthält zu wenige Punkte. Überspringe.");
                continue;
            }

            // Berechne die Bounding Box
            Eigen::Vector4f centroid;
            Eigen::Matrix3f covariance;
            pcl::compute3DCentroid(*cluster_cloud, centroid);
            pcl::computeCovarianceMatrixNormalized(*cluster_cloud, centroid, covariance);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance);
            if (eigen_solver.info() != Eigen::Success) {
                RCLCPP_WARN(this->get_logger(), "Fehler bei der Berechnung der Eigenwerte. Überspringe Cluster.");
                continue;
            }

            Eigen::Matrix3f eigens = eigen_solver.eigenvectors();

            Eigen::Vector4f min_point, max_point;
            pcl::getMinMax3D(*cluster_cloud, indices.indices, min_point, max_point);

            // Marker für RViz erstellen
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = msg->header.frame_id;
            marker.header.stamp = this->now();
            marker.ns = "object_detection";
            marker.id = id++;
            marker.type = visualization_msgs::msg::Marker::CUBE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose.position.x = centroid[0];
            marker.pose.position.y = centroid[1];
            marker.pose.position.z = centroid[2];

            Eigen::Quaternionf quaternion(eigens);
            marker.pose.orientation.x = quaternion.x();
            marker.pose.orientation.y = quaternion.y();
            marker.pose.orientation.z = quaternion.z();
            marker.pose.orientation.w = quaternion.w();

            marker.scale.x = max_point.x() - min_point.x();
            marker.scale.y = max_point.y() - min_point.y();
            marker.scale.z = max_point.z() - min_point.z();

            marker.color.r = 0.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 0.5f;

            marker_array.markers.push_back(marker);
        }

        // Veröffentlichung in RViz
        marker_publisher_->publish(marker_array);
        RCLCPP_INFO(this->get_logger(), "Anzahl der erkannten Objekte: %ld", cluster_indices.size());
    }

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscriber_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetectionVisualizerNode>());
    rclcpp::shutdown();
    return 0;
}