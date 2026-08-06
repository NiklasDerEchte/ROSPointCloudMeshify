#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/vtk_io.h>
#include <pcl/surface/poisson.h>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <pcl/surface/organized_fast_mesh.h>
#include <Eigen/Geometry>
#include <mutex>
#include <cmath>

class Float32ArrayPlaneNode : public rclcpp::Node
{
public:
    Float32ArrayPlaneNode()
        : Node("float_array_unity_plane_node"),
          distance_threshold_(declare_parameter("distance_threshold", 0.01)),
          search_radius_(declare_parameter("search_radius", 0.1)),
          max_neighbors_(declare_parameter("max_neighbors", 150)),
          normal_k_search_(declare_parameter("normal_k_search", 20)),
          output_topic_(declare_parameter("output_topic", "/object_mesh/vertices")),
          plane_topic_(declare_parameter("plane_topic", "/unity/clipping_plane")),
          mode_(declare_parameter("mode", "fast")), // fast, greedy, poisson
          // Unity plane local +Y is the surface normal (default Unity Plane)
          invert_plane_normal_(declare_parameter("invert_plane_normal", false))
    {

        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points/xyzrgba", 100,
            std::bind(&Float32ArrayPlaneNode::pointCloudCallback, this, std::placeholders::_1));

        plane_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            plane_topic_, 10,
            std::bind(&Float32ArrayPlaneNode::planePoseCallback, this, std::placeholders::_1));

        // Interleaved xyzrgba floats for Unity (one message, TRIANGLE_LIST order)
        mesh_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(this->output_topic_, 10);
        RCLCPP_INFO(this->get_logger(),
                    "Node initialized: mesh=%s plane=%s",
                    output_topic_.c_str(), plane_topic_.c_str());
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        try
        {
            RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message");

            bool hasRgb = false;
            bool hasRgba = false;

            for (const auto &field : msg->fields)
            {
                if (field.name == "rgb")
                {
                    hasRgb = true;
                    break;
                }
                else if (field.name == "rgba")
                {
                    hasRgba = true;
                    break;
                }
            }

            if (hasRgb || hasRgba)
            {
                RCLCPP_INFO(this->get_logger(), "Handle XYZ-RGB(A) PointCloud");
                this->handlePointCloud<pcl::PointXYZRGB, pcl::Normal, pcl::PointXYZRGBNormal>(msg);
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Handle XYZ PointCloud");
                this->handlePointCloud<pcl::PointXYZ, pcl::Normal, pcl::PointNormal>(msg);
            }
            RCLCPP_INFO(this->get_logger(), "--- frame done ---");
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error processing PointCloud: %s", e.what());
        }
    }

    // ----------------------------------------------------------------------- [Handle PointCloud Message]

    template <typename PointT, typename NormalT, typename PointNormalT>
    void handlePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS2 PointCloud2 to PCL
        typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
        pcl::fromROSMsg(*msg, *cloud);
        pcl::PolygonMesh mesh;

        if (this->mode_ == "fast" && cloud->isOrganized())
        {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, false, false);

            if (cloud_filtered == nullptr || cloud_filtered->empty())
            {
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Create organized triangulation mesh");
            this->createOrganizedTriangulationMesh<PointT>(cloud_filtered, mesh);
        }
        else if (this->mode_ == "poisson")
        {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, true, true); // ToDo ist das hier richtig? weil poisson doch keine unorganized clouds bekommen darf

            if (cloud_filtered == nullptr || cloud_filtered->empty())
            {
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Create unorganized poisson mesh");
            typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals = this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

            RCLCPP_INFO(this->get_logger(), "Create poisson-mesh");
            this->createPoissonMesh<PointNormalT>(cloud_with_normals, mesh);
        }
        else if (this->mode_ == "greedy" || !cloud->isOrganized())
        {
            typename pcl::PointCloud<PointT>::Ptr cloud_filtered = this->planeSegmentation<PointT>(cloud, false, false);

            if (cloud_filtered == nullptr || cloud_filtered->empty())
            {
                RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
                return;
            }

            RCLCPP_INFO(this->get_logger(), "Create unorganized greedy mesh");
            typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals = this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

            RCLCPP_INFO(this->get_logger(), "Create greedy-triangulation-mesh");
            this->createGreedyTriangulationMesh<PointNormalT>(cloud_with_normals, mesh);
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "Unknown mode: %s", this->mode_.c_str());
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Convert mesh to Float32MultiArray (xyzrgba)");
        std_msgs::msg::Float32MultiArray vertex_array;
        convertMeshToFloatArray<PointT>(mesh, vertex_array);

        if (vertex_array.data.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Float32MultiArray empty, skip publish");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "publish %zu floats (%zu vertices)", vertex_array.data.size(), vertex_array.data.size() / 7);
        mesh_pub_->publish(vertex_array);
    }

    // ----------------------------------------------------------------------- [Surface Reconstruction Algos]

    template <typename PointT>
    void createOrganizedTriangulationMesh(typename pcl::PointCloud<PointT>::Ptr &cloud, pcl::PolygonMesh &mesh)
    {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        typename pcl::OrganizedFastMesh<PointT> ofm;
        ofm.setInputCloud(cloud);
        ofm.setTrianglePixelSize(4); // Größe der Pixel für die Nachbarschaft
        ofm.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
        ofm.reconstruct(mesh);
        RCLCPP_INFO(this->get_logger(), "Organized mesh created in %ld ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1);
    }

    template <typename PointNormalT>
    void createGreedyTriangulationMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh)
    {
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
    void createPoissonMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh)
    {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        // Perform Poisson Surface Reconstruction
        typename pcl::Poisson<PointNormalT> poisson;
        poisson.setDepth(8);             // Depth of reconstruction
        poisson.setSamplesPerNode(1.0f); // Samples per node
        poisson.setSolverDivide(8);      // Solver divide
        poisson.setIsoDivide(8);         // Iso divide
        // poisson.setUsePredictedNormals(true); // Use predicted normals if available
        poisson.setInputCloud(cloud_with_normals);

        poisson.reconstruct(mesh);

        // ToDo Funktioniert das hier trotz Warnung?
        if constexpr (std::is_same<PointNormalT, pcl::PointXYZRGBNormal>::value)
        { // Workaround weil poisson reconstruction Farbinformationen löscht
            typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
            pcl::copyPointCloud<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>(*cloud_with_normals, *cloud_rgb);

            // Rekonstruiere die Punktwolke aus dem Mesh
            pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
            pcl::fromPCLPointCloud2(mesh.cloud, *mesh_cloud);

            // Übertrage RGB-Werte auf rekonstruierte Punkte
            for (size_t i = 0; i < mesh_cloud->points.size(); ++i)
            {
                if (i < cloud_rgb->points.size())
                {
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

    // ----------------------------------------------------------------------- [Convert Mesh -> Float32MultiArray xyzrgba]

    template <typename PointT>
    void convertMeshToFloatArray(
        const pcl::PolygonMesh &mesh,
        std_msgs::msg::Float32MultiArray &vertex_array)
    {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        typename pcl::PointCloud<PointT> cloud;
        pcl::fromPCLPointCloud2(mesh.cloud, cloud);

        if (mesh.polygons.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Triangulation failed: no polygons created");
            return;
        }

        size_t tri = 0;
        for (const auto &p : mesh.polygons)
        {
            if (p.vertices.size() == 3)
            {
                ++tri;
            }
        }

        vertex_array.data.resize(tri * 3 * 7);
        float *d = vertex_array.data.data();
        size_t w = 0;
        for (const auto &polygon : mesh.polygons)
        {
            if (polygon.vertices.size() != 3)
            {
                continue;
            }

            for (const auto vertex_idx : polygon.vertices)
            {
                const auto &point = cloud.points[vertex_idx];

                d[w + 0] = point.x;
                d[w + 1] = point.y;
                d[w + 2] = point.z;

                if constexpr (std::is_same_v<PointT, pcl::PointXYZRGB> ||
                              std::is_same_v<PointT, pcl::PointXYZRGBA>)
                {
                    d[w + 3] = point.r / 255.0f;
                    d[w + 4] = point.g / 255.0f;
                    d[w + 5] = point.b / 255.0f;
                    d[w + 6] = 1.0f;
                }
                else
                {
                    d[w + 3] = 0.0f;
                    d[w + 4] = 1.0f;
                    d[w + 5] = 0.0f;
                    d[w + 6] = 1.0f;
                }
                w += 7;
            }
        }

        RCLCPP_INFO(this->get_logger(), "Mesh converted to Float32MultiArray in %ld ms (%zu vertices)",
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1,
                    vertex_array.data.size() / 7);
    }

    // ----------------------------------------------------------------------- [Helper functions]

    void planePoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        if (msg == nullptr)
        {
            return;
        }
        std::lock_guard<std::mutex> lock(plane_mutex_);
        plane_pose_ = msg->pose;
        has_plane_pose_ = true;
    }

    /// Returns false if no Unity plane pose received yet.
    bool getUnityPlane(Eigen::Vector3d &point_on_plane, Eigen::Vector3d &normal) const
    {
        geometry_msgs::msg::Pose pose;
        {
            std::lock_guard<std::mutex> lock(plane_mutex_);
            if (!has_plane_pose_)
            {
                return false;
            }
            pose = plane_pose_;
        }

        point_on_plane = Eigen::Vector3d(
            pose.position.x,
            pose.position.y,
            pose.position.z);

        // Unity Plane default surface normal is local +Y; pose is already in ROS frame
        Eigen::Quaterniond q(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);
        q.normalize();
        normal = q * Eigen::Vector3d::UnitY();
        if (normal.norm() < 1e-9)
        {
            return false;
        }
        normal.normalize();
        if (invert_plane_normal_)
        {
            normal = -normal;
        }
        return true;
    }

    template <typename PointT, typename NormalT, typename PointNormalT>
    typename pcl::PointCloud<PointNormalT>::Ptr estimateNormals(typename pcl::PointCloud<PointT>::Ptr &cloud)
    {
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
     * Filter using Unity clipping plane.
     * Points on or "under" the plane (signed distance <= distance_threshold_)
     * are treated like former plane inliers: NaN (organized) or removed (unorganized).
     *
     * Normal = rotated Unity +Y. Signed distance = n · (p - plane_point).
     * "Under/on" = signed_distance <= distance_threshold_.
     * Toggle invert_plane_normal if the wrong half-space is filtered.
     *
     * removePlanePoints / removeNaNValues can turn an organized cloud unorganized.
     */
    template <typename PointT>
    typename pcl::PointCloud<PointT>::Ptr planeSegmentation(
        typename pcl::PointCloud<PointT>::Ptr &cloudIn,
        bool removePlanePoints,
        bool removeNaNValues)
    {
        long t1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        typename pcl::PointCloud<PointT>::Ptr cloudOut(new pcl::PointCloud<PointT>());
        pcl::copyPointCloud<PointT, PointT>(*cloudIn, *cloudOut);

        Eigen::Vector3d plane_point;
        Eigen::Vector3d plane_normal;
        if (!getUnityPlane(plane_point, plane_normal))
        {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "No Unity plane pose on %s yet — skip plane filter", plane_topic_.c_str());
            return cloudOut;
        }

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        inliers->indices.reserve(cloudOut->points.size());

        const double thr = distance_threshold_;
        for (size_t i = 0; i < cloudOut->points.size(); ++i)
        {
            const auto &pt = cloudOut->points[i];
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            {
                continue;
            }

            // signed distance to plane; <= thr => on or under (relative to normal)
            const Eigen::Vector3d p(pt.x, pt.y, pt.z);
            const double signed_dist = plane_normal.dot(p - plane_point);
            if (signed_dist <= thr)
            {
                inliers->indices.push_back(static_cast<int>(i));
            }
        }

        if (inliers->indices.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Unity plane filter: no points on/under plane");
        }

        if (removePlanePoints)
        {
            // Unorganized path: drop points on/under plane, keep the rest
            pcl::ExtractIndices<PointT> extract;
            extract.setInputCloud(cloudOut);
            extract.setIndices(inliers);
            extract.setNegative(true);
            extract.filter(*cloudOut);

            if (removeNaNValues)
            {
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*cloudOut, *cloudOut, indices);
            }
        }
        else
        {
            // Organized path: NaN-out points on/under plane (keep image structure)
            for (int idx : inliers->indices)
            {
                cloudOut->points[idx].x = std::numeric_limits<float>::quiet_NaN();
                cloudOut->points[idx].y = std::numeric_limits<float>::quiet_NaN();
                cloudOut->points[idx].z = std::numeric_limits<float>::quiet_NaN();
            }
            if (removeNaNValues)
            {
                std::vector<int> indices;
                pcl::removeNaNFromPointCloud(*cloudOut, *cloudOut, indices);
            }
        }

        RCLCPP_INFO(this->get_logger(),
                    "Unity plane filter in %ld ms (inliers=%zu, n=[%.2f,%.2f,%.2f], p=[%.2f,%.2f,%.2f])",
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t1,
                    inliers->indices.size(),
                    plane_normal.x(), plane_normal.y(), plane_normal.z(),
                    plane_point.x(), plane_point.y(), plane_point.z());
        return cloudOut;
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr plane_sub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr mesh_pub_;

    mutable std::mutex plane_mutex_;
    bool has_plane_pose_{false};
    geometry_msgs::msg::Pose plane_pose_;

    double distance_threshold_;
    double search_radius_;
    int max_neighbors_;
    int normal_k_search_;
    std::string output_topic_;
    std::string plane_topic_;
    std::string mode_;
    bool invert_plane_normal_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Float32ArrayPlaneNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
