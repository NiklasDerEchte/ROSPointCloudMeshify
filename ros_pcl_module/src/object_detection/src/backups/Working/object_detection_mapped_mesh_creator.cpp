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
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <visualization_msgs/msg/marker_array.hpp>

class ObjectDetectionMeshCreatorNode : public rclcpp::Node {
public:
    ObjectDetectionMeshCreatorNode() : Node("object_detection_mesh_creator_node") {
        // Subscriber: Empfang von PointCloud2
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points/xyzrgba", 
            10, 
            std::bind(
                &ObjectDetectionMeshCreatorNode::pointCloudCallback, 
                this, 
                std::placeholders::_1
            )
        );

        // Publisher: Veröffentlichen des Meshes
        mesh_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/object_markers", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Konvertierung von ROS2 PointCloud2 zu PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        // Mapping der Punktwolke
        pcl::PointCloud<pcl::PointXYZ>::Ptr mapped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        
        // ToDo
        // mapPointCloud(cloud, mapped_cloud);
        // pcl::copyPointCloud(*cloud, *mapped_cloud); 

        // 1. Ebene entfernen (Tisch entfernen)
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.01);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.empty()) {
            RCLCPP_WARN(this->get_logger(), "Keine Ebene gefunden.");
            return;
        }

        // Entfernen der Tisch-Ebene aus der Punktwolke
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(true); // Behalte die Punkte, die nicht zur Ebene gehören
        extract.filter(*cloud_filtered);

        // 2. Einschränkung auf relevante Höhen (Objekte auf dem Tisch)
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(coefficients->values[3] + 0.02, coefficients->values[3] + 0.5); // 2cm über der Tischfläche bis 50cm
        pass.filter(*cloud_filtered);

        // Überprüfen, ob die Punktwolke leer ist
        if (cloud_filtered->empty()) {
            RCLCPP_WARN(this->get_logger(), "Die gefilterte Punktwolke ist leer. Verarbeitung wird übersprungen.");
            return;
        }

        // 3. Normalenschätzung
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_xyz(new pcl::search::KdTree<pcl::PointXYZ>());
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud_filtered);
        ne.setSearchMethod(tree_xyz);
        ne.setKSearch(20);
        ne.compute(*normals);

        // Überprüfen, ob Normalen berechnet wurden
        if (normals->empty()) {
            RCLCPP_WARN(this->get_logger(), "Keine Normalen berechnet. Verarbeitung wird übersprungen.");
            return;
        }

        // 4. Mesherstellung mit Greedy Triangulation
        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>());
        pcl::concatenateFields(*cloud_filtered, *normals, *cloud_with_normals);

        pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
        pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
        pcl::PolygonMesh mesh;
        gp3.setSearchRadius(0.05);
        gp3.setMu(2.5);
        gp3.setMaximumNearestNeighbors(100);
        gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 Grad
        gp3.setMinimumAngle(M_PI / 18);       // 10 Grad
        gp3.setMaximumAngle(2 * M_PI / 3);   // 120 Grad
        gp3.setNormalConsistency(false);

        gp3.setInputCloud(cloud_with_normals);
        gp3.setSearchMethod(tree);
        gp3.reconstruct(mesh);

        // 5. Mesh als Marker veröffentlichen
        visualization_msgs::msg::MarkerArray marker_array;
        convertMeshToMarkers(mesh, marker_array);
        mesh_pub_->publish(marker_array);
    }

    void mapPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& input_cloud, 
                       pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud) {
        static pcl::PointCloud<pcl::PointXYZ>::Ptr global_map(new pcl::PointCloud<pcl::PointXYZ>());
        static Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity(); // Dummy Transformation (ersetzbar)

        // Transformiere die eingehende Punktwolke
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::transformPointCloud(*input_cloud, *transformed_cloud, transformation);

        // Füge die transformierte Punktwolke zur globalen Karte hinzu
        *global_map += *transformed_cloud;

        // Wende einen VoxelGrid-Filter an, um die Punktwolke zu vereinfachen
        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setInputCloud(global_map);
        voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f);
        voxel_grid.filter(*output_cloud);

        // Überprüfen, ob die Punktwolke leer ist
        if (output_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Die transformierte und gefilterte Punktwolke ist leer.");
        }

        // Aktualisiere die globale Karte
        global_map = output_cloud;
    }

    void convertMeshToMarkers(const pcl::PolygonMesh &mesh, visualization_msgs::msg::MarkerArray &marker_array) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "zivid_optical_frame";
        marker.header.stamp = this->now();
        marker.ns = "mesh";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;

        // Konvertierung der PolygonMesh-Daten in Marker-Geometrie
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
    rclcpp::spin(std::make_shared<ObjectDetectionMeshCreatorNode>());
    rclcpp::shutdown();
    return 0;
}
