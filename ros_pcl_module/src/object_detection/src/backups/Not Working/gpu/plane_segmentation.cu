#include <cuda_runtime.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_types.h>

// CUDA-Kernel zum Entfernen der Ebenenpunkte
__global__ void removePlanePointsGPU(pcl::PointXYZ* cloud, int* inliers, int numInliers, int totalPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPoints) return;

    for (int i = 0; i < numInliers; i++) {
        if (idx == inliers[i]) {
            cloud[idx].x = NAN;
            cloud[idx].y = NAN;
            cloud[idx].z = NAN;
            return;
        }
    }
}

// Wrapper-Funktion für den CUDA-Kernel
void launchRemovePlanePointsKernel(pcl::gpu::DeviceArray<pcl::PointXYZ>& cloud_device, pcl::gpu::DeviceArray<int>& inliers_gpu) {
    int totalPoints = cloud_device.size();
    int numInliers = inliers_gpu.size();

    int threadsPerBlock = 256;
    int numBlocks = (totalPoints + threadsPerBlock - 1) / threadsPerBlock;

    removePlanePointsGPU<<<numBlocks, threadsPerBlock>>>(
        cloud_device.ptr(), inliers_gpu.ptr(), numInliers, totalPoints
    );

    cudaDeviceSynchronize();
}
