#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include "pcl/features/normal_3d_omp.h"
#include <pcl/visualization/pcl_visualizer.h>
#include "pcl/filters/passthrough.h"
#include <chrono>
#include <ctime>
#include "pcl/common/transforms.h"
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/features/features.hpp>

void GetTranform(const std::vector<float> eul, Eigen::Matrix4f& transform){
    float x = eul.at(0);
    float y = eul.at(1);
    float z = eul.at(2);
    transform<< cos(y)*cos(z), sin(x)*sin(y)*cos(z) - cos(x)*sin(z), cos(x)*sin(y)*cos(z) + sin(x)*sin(z), 0.0,
                cos(y)*sin(z), sin(x)*sin(y)*sin(z) + cos(x)*cos(z), cos(x)*sin(y)*sin(z) - sin(x)*cos(z), 0.0,
                -sin(y),       sin(x)*cos(y),                        cos(x)*cos(y),                        0.0,
                0.0,           0.0,                                  0.0,                                  1.0;
}

void NormalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr& pass_filtered_cloud,
        std::vector<pcl::PointXYZ>& h_normals, pcl::PointCloud<pcl::PointXYZ>::Ptr& gpu_normal_filter_cloud){
    for(std::size_t i = 0;i < h_normals.size(); ++i){
        pcl::PointXYZ xyz = h_normals[i];
        float curvature = xyz.data[3];
        auto nor_x = xyz.x;
        auto nor_y = xyz.y;
        auto nor_z = xyz.z;
        auto p_x = pass_filtered_cloud->at(i).x;
        auto p_y = pass_filtered_cloud->at(i).y;
        auto p_z = pass_filtered_cloud->at(i).z;
        float mul = nor_x*p_x + nor_y*p_y + nor_z*p_z;
        float nor_len = std::sqrt(nor_x*nor_x + nor_y*nor_y + nor_z*nor_z);
        float p_len = std::sqrt(p_x*p_x + p_y*p_y + p_z*p_z);
        float theta_cos = mul/nor_len/p_len;
        if(std::abs(theta_cos) > 0.6 && curvature < 0.2) {
            gpu_normal_filter_cloud->push_back(pass_filtered_cloud->at(i));
        }
    }
}

void NormalFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr& pass_filtered_cloud,
        pcl::PointCloud<pcl::Normal>::Ptr& normals, pcl::PointCloud<pcl::PointXYZ>::Ptr& cpu_normal_filter_cloud){
    for(int i = 0;i<normals->size();i++){
        auto nor_x = normals->at(i).normal_x;
        auto nor_y = normals->at(i).normal_y;
        auto nor_z = normals->at(i).normal_z;
        auto p_x = pass_filtered_cloud->at(i).x;
        auto p_y = pass_filtered_cloud->at(i).y;
        auto p_z = pass_filtered_cloud->at(i).z;
        float mul = nor_x*p_x + nor_y*p_y + nor_z*p_z;
        float nor_len = std::sqrt(nor_x*nor_x + nor_y*nor_y + nor_z*nor_z);
        float p_len = std::sqrt(p_x*p_x + p_y*p_y + p_z*p_z);
        float theta_cos = mul/nor_len/p_len;
        if(std::abs(theta_cos) > 0.6 && normals->at(i).curvature < 0.2) {
            cpu_normal_filter_cloud->push_back(pass_filtered_cloud->at(i));
        }
    }
}

void ShowPointCloud(pcl::visualization::PCLVisualizer& viewer,pcl::PointCloud<pcl::PointXYZ>::Ptr& trans_cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& pass_filtered_cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr& normal_filter_cloud,
                    pcl::PointCloud<pcl::PointXYZ>::Ptr& gpu_normal_filter_cloud){
    int v1(0), v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_h(trans_cloud, 255, 255, 255);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pass_h(pass_filtered_cloud, 255, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> filtered_h(normal_filter_cloud, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> gpu_filtered_h(gpu_normal_filter_cloud, 0, 255, 0);
    viewer.addPointCloud(trans_cloud, cloud_h, "original cloud", v1);
    viewer.addPointCloud(trans_cloud, cloud_h, "original cloud2", v2);
    viewer.addPointCloud(pass_filtered_cloud, pass_h, "pass cloud", v1);
    viewer.addPointCloud(pass_filtered_cloud, pass_h, "pass cloud2", v2);
    viewer.addPointCloud(normal_filter_cloud, filtered_h, "filtered cloud", v1);
    viewer.addPointCloud(gpu_normal_filter_cloud, gpu_filtered_h, "filtered cloud2", v2);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "pass cloud", 0);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "filtered cloud", 0);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "pass cloud2", 1);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "filtered cloud2", 1);
}
int main()
{
    //读取点云数据
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::string file_path = "..//data//data.pcd";
    pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud);

    //点云预变换
    pcl::PointCloud<pcl::PointXYZ>::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    std::vector<float> eul{1.5514,  -0.1149,  -3.1405}; //将输入点云的地面与坐标系XY平面对齐
    GetTranform(eul, transform);
    pcl::transformPointCloud(*cloud, *trans_cloud, transform);

    //直通滤波器, 滤除Z轴方向上的部分点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr pass_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud (trans_cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (-0.05, 0.3);
    pass.filter (*pass_filtered_cloud);

    //GPU计算法线
    pcl::gpu::NormalEstimation::Normals d_normal;
    pcl::gpu::NormalEstimation::PointCloud d_cloud;
    d_cloud.upload(pass_filtered_cloud->points);
    pcl::gpu::NormalEstimation normal_estimator;
    normal_estimator.setInputCloud(d_cloud);
    normal_estimator.setRadiusSearch(0.2, d_cloud.size());
    normal_estimator.compute(d_normal);
    std::vector<pcl::PointXYZ> h_normals;
    d_normal.download(h_normals);
    pcl::PointCloud<pcl::PointXYZ>::Ptr gpu_normal_filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    NormalFilter(pass_filtered_cloud, h_normals, gpu_normal_filter_cloud);

    //CPU计算法线
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(pass_filtered_cloud);
    n.setInputCloud(pass_filtered_cloud);
    n.setSearchMethod(tree);
//    n.setKSearch(100);
    n.setRadiusSearch(0.2);
    n.compute(*normals);
    //根据法线和视线进行滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr normal_filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    NormalFilter(pass_filtered_cloud, normals, normal_filter_cloud);

    //可视化
    pcl::visualization::PCLVisualizer viewer("filtered point Viewer");
    ShowPointCloud(viewer, trans_cloud, pass_filtered_cloud, normal_filter_cloud, gpu_normal_filter_cloud);
    while (!viewer.wasStopped())
    {
        viewer.spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
    return 0;
}