#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter.h>                 // für removeNaNFromPointCloud
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/vtk_io.h>
#include <pcl/surface/poisson.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/surface/organized_fast_mesh.h>
#include <chrono>
#include <limits>

class MarkerArrayNode : public rclcpp::Node {
public:
  MarkerArrayNode()
    : Node("marker_array_node_2"),
      distance_threshold_(declare_parameter("distance_threshold", 0.01)),
      search_radius_(declare_parameter("search_radius", 0.1)),
      max_neighbors_(declare_parameter("max_neighbors", 150)),
      normal_k_search_(declare_parameter("normal_k_search", 20)),
      output_topic_(declare_parameter("output_topic", "/object_markers")),
      mode_(declare_parameter("mode", "fast")) // fast, greedy, poisson
  {
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/points/xyzrgba", 10,
      std::bind(&MarkerArrayNode::pointCloudCallback, this, std::placeholders::_1));

    mesh_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(output_topic_, 10);
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

      if (hasRgb || hasRgba) {
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
    typename pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*msg, *cloud);

    // WICHTIG: NaNs sofort entfernen (verhindert den KdTree-Crash)
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    if (cloud->empty()) {
      RCLCPP_WARN(this->get_logger(), "PointCloud is empty after NaN removal");
      return;
    }

    pcl::PolygonMesh mesh;

    if (this->mode_ == "fast" && cloud->isOrganized()) {
      typename pcl::PointCloud<PointT>::Ptr cloud_filtered =
        this->planeSegmentation<PointT>(cloud, false, false);

      if (cloud_filtered == nullptr || cloud_filtered->empty()) {
        RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
        return;
      }

      RCLCPP_INFO(this->get_logger(), "Create organized triangulation mesh");
      this->createOrganizedTriangulationMesh<PointT>(cloud_filtered, mesh);

    } else if (this->mode_ == "poisson") {
      typename pcl::PointCloud<PointT>::Ptr cloud_filtered =
        this->planeSegmentation<PointT>(cloud, true, true);

      if (cloud_filtered == nullptr || cloud_filtered->empty()) {
        RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
        return;
      }

      // Nochmal NaNs entfernen (sicherheitshalber)
      std::vector<int> nan_indices;
      pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, nan_indices);

      RCLCPP_INFO(this->get_logger(), "Create unorganized poisson mesh");
      typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals =
        this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

      this->createPoissonMesh<PointNormalT>(cloud_with_normals, mesh);

    } else { // greedy oder unorganized
      typename pcl::PointCloud<PointT>::Ptr cloud_filtered =
        this->planeSegmentation<PointT>(cloud, true, true);   // Plane + NaNs entfernen

      if (cloud_filtered == nullptr || cloud_filtered->empty()) {
        RCLCPP_WARN(this->get_logger(), "Filtered PointCloud is empty");
        return;
      }

      // Sicherstellen, dass keine NaNs mehr drin sind
      std::vector<int> nan_indices;
      pcl::removeNaNFromPointCloud(*cloud_filtered, *cloud_filtered, nan_indices);

      RCLCPP_INFO(this->get_logger(), "Create unorganized greedy mesh");
      typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals =
        this->estimateNormals<PointT, NormalT, PointNormalT>(cloud_filtered);

      this->createGreedyTriangulationMesh<PointNormalT>(cloud_with_normals, mesh);
    }

    RCLCPP_INFO(this->get_logger(), "Convert mesh to markers");
    visualization_msgs::msg::MarkerArray marker_array;
    convertMeshToMarkers<PointT>(mesh, marker_array);
    mesh_pub_->publish(marker_array);
  }

  // ----------------------------------------------------------------------- [Surface Reconstruction]
  template <typename PointT>
  void createOrganizedTriangulationMesh(typename pcl::PointCloud<PointT>::Ptr &cloud, pcl::PolygonMesh &mesh) {
    auto t1 = std::chrono::steady_clock::now();

    pcl::OrganizedFastMesh<PointT> ofm;
    ofm.setInputCloud(cloud);
    ofm.setTrianglePixelSize(4);
    ofm.setTriangulationType(pcl::OrganizedFastMesh<PointT>::TRIANGLE_RIGHT_CUT);
    ofm.reconstruct(mesh);

    auto t2 = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(), "Organized mesh created in %ld ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
  }

  template <typename PointNormalT>
  void createGreedyTriangulationMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
    auto t1 = std::chrono::steady_clock::now();

    typename pcl::search::KdTree<PointNormalT>::Ptr tree(new pcl::search::KdTree<PointNormalT>());
    pcl::GreedyProjectionTriangulation<PointNormalT> gp3;
    gp3.setSearchRadius(search_radius_);
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(max_neighbors_);
    gp3.setMaximumSurfaceAngle(M_PI / 4);
    gp3.setMinimumAngle(M_PI / 18);
    gp3.setMaximumAngle(2 * M_PI / 3);
    gp3.setNormalConsistency(false);
    gp3.setInputCloud(cloud_with_normals);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(mesh);

    auto t2 = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(), "Greedy mesh created in %ld ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
  }

  template <typename PointNormalT>
  void createPoissonMesh(typename pcl::PointCloud<PointNormalT>::Ptr &cloud_with_normals, pcl::PolygonMesh &mesh) {
    auto t1 = std::chrono::steady_clock::now();

    pcl::Poisson<PointNormalT> poisson;
    poisson.setDepth(8);
    poisson.setSamplesPerNode(1.0f);
    poisson.setSolverDivide(8);
    poisson.setIsoDivide(8);
    poisson.setInputCloud(cloud_with_normals);
    poisson.reconstruct(mesh);

    // Farben wiederherstellen (Poisson löscht sie)
    if constexpr (std::is_same_v<PointNormalT, pcl::PointXYZRGBNormal>) {
      typename pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr mesh_cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
      pcl::fromPCLPointCloud2(mesh.cloud, *mesh_cloud);

      // Einfaches Nearest-Neighbor Color Transfer wäre besser, hier nur Fallback
      for (size_t i = 0; i < mesh_cloud->size() && i < cloud_with_normals->size(); ++i) {
        mesh_cloud->points[i].r = cloud_with_normals->points[i].r;
        mesh_cloud->points[i].g = cloud_with_normals->points[i].g;
        mesh_cloud->points[i].b = cloud_with_normals->points[i].b;
      }
      pcl::toPCLPointCloud2(*mesh_cloud, mesh.cloud);
    }

    auto t2 = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(), "Poisson mesh created in %ld ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
  }

  // ----------------------------------------------------------------------- [Convert Mesh -> Markers]
  template <typename PointT>
  void convertMeshToMarkers(const pcl::PolygonMesh &mesh, visualization_msgs::msg::MarkerArray &marker_array) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "mesh";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = marker.scale.y = marker.scale.z = 1.0;
    marker.pose.orientation.w = 1.0;

    typename pcl::PointCloud<PointT> cloud;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud);

    if (mesh.polygons.empty()) {
      RCLCPP_WARN(this->get_logger(), "Triangulation failed: no polygons created");
      return;
    }

    for (const auto &polygon : mesh.polygons) {
      if (polygon.vertices.size() != 3) continue;

      for (const auto &vertex_idx : polygon.vertices) {
        const auto &point = cloud.points[vertex_idx];

        geometry_msgs::msg::Point pt;
        pt.x = point.x;
        pt.y = point.y;
        pt.z = point.z;
        marker.points.push_back(pt);

        if constexpr (std::is_same_v<PointT, pcl::PointXYZRGB> || std::is_same_v<PointT, pcl::PointXYZRGBA>) {
          std_msgs::msg::ColorRGBA color;
          color.a = 1.0f;
          color.r = static_cast<float>(point.r) / 255.0f;
          color.g = static_cast<float>(point.g) / 255.0f;
          color.b = static_cast<float>(point.b) / 255.0f;
          marker.colors.push_back(color);
        }
      }
    }

    if constexpr (std::is_same_v<PointT, pcl::PointXYZ>) {
      marker.color.a = 1.0f;
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
    }

    marker_array.markers.push_back(marker);
  }

  // ----------------------------------------------------------------------- [Helper]
  template <typename PointT, typename NormalT, typename PointNormalT>
  typename pcl::PointCloud<PointNormalT>::Ptr estimateNormals(typename pcl::PointCloud<PointT>::Ptr &cloud) {
    auto t1 = std::chrono::steady_clock::now();

    // Nochmal sicherstellen, dass keine NaNs vorhanden sind
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

    typename pcl::PointCloud<NormalT>::Ptr normals(new pcl::PointCloud<NormalT>());
    typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());

    pcl::NormalEstimation<PointT, NormalT> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setKSearch(normal_k_search_);
    ne.compute(*normals);

    typename pcl::PointCloud<PointNormalT>::Ptr cloud_with_normals(new pcl::PointCloud<PointNormalT>());
    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

    auto t2 = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(), "Normals estimated in %ld ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

    return cloud_with_normals;
  }

  template <typename PointT>
  typename pcl::PointCloud<PointT>::Ptr planeSegmentation(
    typename pcl::PointCloud<PointT>::Ptr &cloudIn,
    bool removePlanePoints,
    bool removeNaNValues)
  {
    auto t1 = std::chrono::steady_clock::now();

    typename pcl::PointCloud<PointT>::Ptr cloudOut(new pcl::PointCloud<PointT>());
    pcl::copyPointCloud(*cloudIn, *cloudOut);

    if (removeNaNValues) {
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*cloudOut, *cloudOut, indices);
    }

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
      return cloudOut;   // lieber die Cloud zurückgeben als nullptr
    }

    if (removePlanePoints) {
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(cloudOut);
      extract.setIndices(inliers);
      extract.setNegative(true);
      extract.filter(*cloudOut);
    } else {
      // Für organized meshes: Punkte auf NaN setzen
      for (int idx : inliers->indices) {
        cloudOut->points[idx].x = std::numeric_limits<float>::quiet_NaN();
        cloudOut->points[idx].y = std::numeric_limits<float>::quiet_NaN();
        cloudOut->points[idx].z = std::numeric_limits<float>::quiet_NaN();
      }
    }

    auto t2 = std::chrono::steady_clock::now();
    RCLCPP_INFO(this->get_logger(), "Plane segmentation in %ld ms",
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());

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