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
#include <shape_msgs/msg/mesh.hpp>

class MeshNode : public rclcpp::Node {
public:
    MeshNode() 
        : Node("mesh_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          search_radius_(declare_parameter("search_radius", 0.1)),
          max_neighbors_(declare_parameter("max_neighbors", 150)),
          normal_k_search_(declare_parameter("normal_k_search", 20)),
          output_topic_(declare_parameter("output_topic", "/object_markers")),
          mode_(declare_parameter("mode", "fast")) // fast, greedy, poisson
        {

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/points/xyzrgba", 10, 
            std::bind(&MeshNode::pointCloudCallback, this, std::placeholders::_1)
        );

        mesh_pub_ = this->create_publisher<shape_msgs::msg::Mesh>(this->output_topic_, 10);
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
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, true, true);

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
        shape_msgs::msg::Mesh mesh_msg = convertMeshToShapeMsg<PointT>(mesh);

        RCLCPP_INFO(this->get_logger(), "publish");
        mesh_pub_->publish(mesh_msg);
    }

    // ----------------------------------------------------------------------- [Surface Reconstruction Algos]

    template <typename PointT>
    void createOrganizedTriangulationMesh(typename pcl::PointCloud<PointT>::Ptr &cloud, pcl::PolygonMesh &mesh) {
        typename pcl::OrganizedFastMesh<PointT> ofm;
        ofm.setInputCloud(cloud);
        ofm.setTrianglePixelSize(4); // Größe der Pixel für die Nachbarschaft
        ofm.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
        ofm.reconstruct(mesh);
    }

    template <typename PointNormalT>
    void createGreedyTriangulationMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
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
    }

    template <typename PointNormalT>
    void createPoissonMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
     
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
    }

    // ----------------------------------------------------------------------- [Convert Mesh]

    template <typename PointT>
    shape_msgs::msg::Mesh convertMeshToShapeMsg(const pcl::PolygonMesh &mesh) {
        shape_msgs::msg::Mesh mesh_msg;
        
        typename pcl::PointCloud<PointT> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        // Füge Punkte hinzu
        // ToDo Farben???
        mesh_msg.vertices.resize(cloud.points.size());
        for (size_t i = 0; i < cloud.points.size(); ++i) {
            mesh_msg.vertices[i].x = cloud.points[i].x;
            mesh_msg.vertices[i].y = cloud.points[i].y;
            mesh_msg.vertices[i].z = cloud.points[i].z;
        }

        // Füge Indices der Dreiecke hinzu
        mesh_msg.triangles.resize(mesh.polygons.size());
        for (size_t i = 0; i < mesh.polygons.size(); ++i) {
            const auto &polygon = mesh.polygons[i];
            if (polygon.vertices.size() == 3) { // Nur Dreiecke akzeptieren
                mesh_msg.triangles[i].vertex_indices[0] = polygon.vertices[0];
                mesh_msg.triangles[i].vertex_indices[1] = polygon.vertices[1];
                mesh_msg.triangles[i].vertex_indices[2] = polygon.vertices[2];
            }
        }

        return mesh_msg;
    }

    // ----------------------------------------------------------------------- [Helper functions]

    template <typename PointT, typename NormalT, typename PointNormalT>
    typename pcl::PointCloud<PointNormalT>::Ptr estimateNormals(typename pcl::PointCloud<PointT>::Ptr &cloud) {
        RCLCPP_INFO(this->get_logger(), "Estimate normals");
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

        return cloudOut;
    }


    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<shape_msgs::msg::Mesh>::SharedPtr mesh_pub_;

    double distance_threshold_;
    double search_radius_;
    int max_neighbors_;
    int normal_k_search_;
    std::string output_topic_;
    std::string mode_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MeshNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}