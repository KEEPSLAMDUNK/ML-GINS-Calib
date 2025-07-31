// Main calibration node for multi-LiDAR GNSS/INS extrinsic calibration
#include <filesystem>
#include <iostream>
#include <unordered_set>
#include <yaml-cpp/yaml.h>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "ml_gins_calib/optimization/sensor_data_loader.h"
#include "ml_gins_calib/optimization/extrinsic_calibrator.h"

std::vector<bool> selectKeyframes(const std::vector<Eigen::Isometry3d> &poses,
                                  double translation_threshold,
                                  double rotation_threshold);
std::vector<std::string> filterPathsBySelection(const std::vector<std::string> &paths,
                                               const std::vector<bool> &selection);
std::vector<Eigen::Isometry3d>
filterPosesBySelection(const std::vector<Eigen::Isometry3d> &poses,
                      const std::vector<bool> &selection);
Eigen::Matrix3d rotationMatrixToTangent(const Eigen::Matrix3d &R);
Eigen::Matrix3d tangentToRotationMatrix(const Eigen::Matrix3d &omega);
Eigen::Isometry3d
computeAverageTransform(const std::vector<Eigen::Isometry3d> &transforms);
CloudPtr
buildPointCloudMap(const std::vector<std::string> &cloud_paths,
                   const std::vector<Eigen::Isometry3d> &transforms,
                   const Eigen::Isometry3d T_ext = Eigen::Isometry3d::Identity());

double analyzeTerrainRoughness(const CloudPtr &cloud,
                              const std::vector<Eigen::Isometry3d> &poses);

int main() {
  using namespace calib;
  namespace fs = std::filesystem;

  std::string config_path = "./config/params.yaml";

  DataLoader::SensorList sensor_list;
  DataLoader::LoadSensorList(config_path, sensor_list);

  DataLoader data_loader(sensor_list);
  data_loader.LoadLtgExtInfo(config_path);
  data_loader.LoadCalibData();
  data_loader.LoadGinsInstallationHeight(config_path);

  // Load optimization parameters from config
  YAML::Node config_node = YAML::LoadFile(config_path);
  YAML::Node opt_params = config_node["optimization_params"];
  
  int num_threads = opt_params["num_threads"].as<int>();
  int lg_max_iter = opt_params["lidar_gins_optimization"]["max_iterations"].as<int>();
  double lg_max_corr_dist = opt_params["lidar_gins_optimization"]["max_correspondence_distance"].as<double>();
  int ml_max_iter = opt_params["multi_lidar_optimization"]["max_iterations"].as<int>();
  double ml_max_corr_dist = opt_params["multi_lidar_optimization"]["max_correspondence_distance"].as<double>();
  double rotation_threshold = opt_params["keyframe_selection"]["rotation_threshold"].as<double>();
  double translation_threshold = opt_params["keyframe_selection"]["translation_threshold"].as<double>();
  int time_precision = opt_params["output"]["time_precision"].as<int>();
  double time_scale_factor = opt_params["output"]["time_scale_factor"].as<double>();
  std::string output_file = opt_params["output"]["extrinsic_output_file"].as<std::string>();

  auto ext_infos = data_loader.ltg_ext_infos_;
  auto T_vec = data_loader.gnss_ins_data_;
  data_loader.SetTrajectoryOrigin(T_vec[0], T_vec);

  auto selection = selectKeyframes(T_vec, translation_threshold, rotation_threshold);

  std::cout << "LiDAR-GNSS/INS Extrinsic Calibration" << std::endl;

  std::vector<std::pair<CloudPtr, CloudPtr>> lg_map_arr;
  std::vector<CloudPtr> lidar_align_map_arr;

  std::vector<std::vector<Eigen::Isometry3d>> T_exts_arr;
  std::vector<Eigen::Isometry3d> gins_poses;

  for (size_t i = 0; i < ext_infos.size(); ++i) {
    std::cout << "########## " << ext_infos[i].lidar_name_ << " ##########"
              << std::endl;

    std::vector<Eigen::Isometry3d> T_exts;
    T_exts.emplace_back(ext_infos[i].T_ext_);

    std::vector<std::string> cloud_file_path_arr =
        data_loader.lidar_data_list_[i].data_list_;

    auto T_vec_selection = filterPosesBySelection(T_vec, selection);
    cloud_file_path_arr = filterPathsBySelection(cloud_file_path_arr, selection);

    std::cout << "Data size: " << cloud_file_path_arr.size() << std::endl;

    auto start = std::chrono::steady_clock::now();

    ExtrinsicCalibrator opt;
    opt.SetNumThreads(num_threads);
    opt.SetOptIter(lg_max_iter);
    opt.SetMaxCorrDis(lg_max_corr_dist);
    opt.SetGinsHeight(data_loader.gins_installation_height_);

    CloudPtr map_lidar_align =
        buildPointCloudMap(cloud_file_path_arr, T_vec_selection, ext_infos[i].T_ext_);
    lidar_align_map_arr.push_back(map_lidar_align);

    auto T_ext_opt = opt.LgOptimizationCoarse(
        cloud_file_path_arr, T_vec_selection, ext_infos[i].T_ext_);

    ext_infos[i].T_ext_ = T_ext_opt;

    CloudPtr map_ori =
        buildPointCloudMap(cloud_file_path_arr, T_vec_selection, ext_infos[i].T_ext_);

    T_ext_opt = opt.LgOptimizationRefine(cloud_file_path_arr, T_vec_selection,
                                         T_ext_opt);
    ext_infos[i].T_ext_ = T_ext_opt;

    T_exts.insert(T_exts.end(), opt.T_exts_.begin(), opt.T_exts_.end());
    T_exts_arr.emplace_back(T_exts);

    CloudPtr map_opt =
        buildPointCloudMap(cloud_file_path_arr, T_vec_selection, ext_infos[i].T_ext_);

    if (i == 0)
      gins_poses = T_vec_selection;

    lg_map_arr.push_back(std::make_pair(map_ori, map_opt));

    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << std::setprecision(time_precision) << std::fixed;
    std::cout << "Average running time: " << duration / time_scale_factor << "s"
              << std::endl
              << std::endl;
  }

  std::cout << "Ground Alignment" << std::endl;

//  CloudPtr target_cloud(new PointCloudType);
//  pcl::io::loadPCDFile(data_loader.lidar_data_list_[0].data_list_[0],
//                       *target_cloud);
//  pcl::transformPointCloud(*target_cloud, *target_cloud,
//                           ext_infos[0].T_ext_.matrix());
//  for (size_t i = 1; i < ext_infos.size(); ++i) {
//    CloudPtr source_cloud(new PointCloudType);
//    pcl::io::loadPCDFile(data_loader.lidar_data_list_[i].data_list_[0],
//                         *source_cloud);
//    pcl::transformPointCloud(*source_cloud, *source_cloud,
//                             ext_infos[i].T_ext_.matrix());
//
//    ExtrinsicCalibrator opt;
//    auto T_ga = opt.GroundAlignment(source_cloud, target_cloud);
//    ext_infos[i].T_ext_ = T_ga * ext_infos[i].T_ext_;
//  }

  std::cout << std::endl << std::endl;

  std::cout << "Multi-LiDAR Extrinsic Calibration" << std::endl;

  std::pair<CloudPtr, CloudPtr> ml_map;

  ExtrinsicCalibrator opt;
  opt.SetNumThreads(num_threads);
  opt.SetOptIter(ml_max_iter);
  opt.SetMaxCorrDis(ml_max_corr_dist);

  std::vector<std::string> multi_lidar_cloud_paths;
  std::vector<std::vector<std::string>> multi_lidar_cloud_paths2;
  std::vector<Eigen::Isometry3d> multi_lidar_poses;
  for (size_t i = 0; i < ext_infos.size(); ++i) {
    std::vector<std::string> cloud_file_path_arr =
        data_loader.lidar_data_list_[i].data_list_;

    auto T_vec_selection = filterPosesBySelection(T_vec, selection);
    cloud_file_path_arr = filterPathsBySelection(cloud_file_path_arr, selection);

    multi_lidar_cloud_paths.insert(multi_lidar_cloud_paths.end(),
                                   cloud_file_path_arr.begin(),
                                   cloud_file_path_arr.end());
    multi_lidar_cloud_paths2.emplace_back(cloud_file_path_arr);

    for (size_t j = 0; j < T_vec_selection.size(); ++j) {
      multi_lidar_poses.emplace_back(T_vec_selection[j] * ext_infos[i].T_ext_);
    }
  }

  std::vector<Eigen::Isometry3d> ltl_exts(ext_infos.size());
  for (size_t i = 0; i < ext_infos.size(); ++i) {
    ltl_exts[i] = ext_infos[0].T_ext_.inverse() * ext_infos[i].T_ext_;

    Eigen::Vector3d t = ltl_exts[i].translation();
    Eigen::Quaterniond q(ltl_exts[i].rotation());
    std::cout << "ltl_exts " << i << " : " << t.transpose() << " "
              << q.coeffs().transpose() << std::endl;
  }

  CloudPtr ml_map_ori =
      buildPointCloudMap(multi_lidar_cloud_paths, multi_lidar_poses);

  auto start = std::chrono::steady_clock::now();

  std::vector<Eigen::Isometry3d> opt_poses;
  opt.JointOptimization(multi_lidar_cloud_paths2, gins_poses, ltl_exts,
                        ext_infos[0].T_ext_);

  for (size_t i = 0; i < ext_infos.size(); ++i) {
    for (size_t j = 0; j < gins_poses.size(); ++j) {
      opt_poses.emplace_back(gins_poses[j] * ltl_exts[i]);
    }
  }

  CloudPtr ml_map_opt = buildPointCloudMap(multi_lidar_cloud_paths, opt_poses);

  analyzeTerrainRoughness(ml_map_opt, opt_poses);

  ml_map.first = ml_map_ori;
  ml_map.second = ml_map_opt;

  std::vector<Eigen::Isometry3d> multi_lidar_exts = ltl_exts;
  for (size_t i = 0; i < ext_infos.size(); ++i) {
    std::cout << ext_infos[i].lidar_name_ << std::endl;
    Eigen::Vector3d t = multi_lidar_exts[i].translation();
    Eigen::Quaterniond q(multi_lidar_exts[i].rotation());
    std::cout << t.x() << " " << t.y() << " " << t.z() << std::endl
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w()
              << std::endl
              << std::endl;
  }

  auto end = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  std::cout << std::setprecision(time_precision) << std::fixed;
  std::cout << "Average running time: " << duration / time_scale_factor << "s"
            << std::endl;



  // save multi-lidar extrinsic parameters
  std::ofstream outfile(output_file);
  for (const auto &T_ext : multi_lidar_exts) {
    Eigen::Quaterniond q(T_ext.rotation());
    Eigen::Vector3d t(T_ext.translation());

    outfile << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
            << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  outfile.close();

  // save LiDAR-GINS/INS calibration process
  //  for (size_t i = 0; i < T_exts_arr.size(); ++i) {
  //    outfile.open("lidar_gins" + std::to_string(i) + ".txt");
  //    for (size_t j = 0; j < T_exts_arr[i].size(); ++j) {
  //      Eigen::Quaterniond q(T_exts_arr[i][j].rotation());
  //      Eigen::Vector3d t(T_exts_arr[i][j].translation());
  //
  //      outfile << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << "
  //      "
  //              << q.y() << " " << q.z() << " " << q.w() << std::endl;
  //    }
  //    outfile.close();
  //  }

  return 0;
}

std::vector<bool> selectKeyframes(const std::vector<Eigen::Isometry3d> &poses,
                                  double translation_threshold,
                                  double rotation_threshold) {
  std::vector<bool> selection(poses.size(), false);
  selection[0] = true;
  float t_travel = 0;
  float R_travel = 0;
  Eigen::Vector3d last_position = poses[0].translation();
  Eigen::Matrix3d last_rotation = poses[0].rotation();
  for (size_t i = 1; i < poses.size(); ++i) {
    // translation
    t_travel += (poses[i].translation() - last_position).norm();
    last_position = poses[i].translation();
    if (t_travel >= translation_threshold) {
      selection[i] = true;
      t_travel = 0;
      R_travel = 0;
    }

    // rotation
    Eigen::Matrix3d dR = last_rotation.transpose() * poses[i].rotation();
    Eigen::AngleAxisd dR_aa(dR);
    R_travel += dR_aa.angle();
    last_rotation = poses[i].rotation();
    if (R_travel >= rotation_threshold) {
      selection[i] = true;
      R_travel = 0;
      t_travel = 0;
    }
  }

  return selection;
}

std::vector<std::string> filterPathsBySelection(const std::vector<std::string> &paths,
                                               const std::vector<bool> &selection) {
  std::vector<std::string> selected_paths;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (selection[i]) {
      selected_paths.push_back(paths[i]);
    }
  }
  return selected_paths;
}

std::vector<Eigen::Isometry3d>
filterPosesBySelection(const std::vector<Eigen::Isometry3d> &poses,
                      const std::vector<bool> &selection) {
  std::vector<Eigen::Isometry3d> selected_poses;
  for (size_t i = 0; i < poses.size(); ++i) {
    if (selection[i]) {
      selected_poses.push_back(poses[i]);
    }
  }
  return selected_poses;
}

Eigen::Matrix3d rotationMatrixToTangent(const Eigen::Matrix3d &R) {
  double theta = acos((R.trace() - 1) / 2);
  Eigen::Matrix3d omega = theta / (2 * sin(theta)) * (R - R.transpose());
  return omega;
}

Eigen::Matrix3d tangentToRotationMatrix(const Eigen::Matrix3d &omega) {
  double theta = sqrt(omega(2, 1) * omega(2, 1) + omega(0, 2) * omega(0, 2) +
                      omega(1, 0) * omega(1, 0));
  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R = I + (sin(theta) / theta) * omega +
                      (1 - cos(theta)) / (theta * theta) * (omega * omega);
  return R;
}

Eigen::Isometry3d
computeAverageTransform(const std::vector<Eigen::Isometry3d> &transforms) {
  if (transforms.empty()) {
    // Return identity if no transforms provided.
    return Eigen::Isometry3d::Identity();
  }

  // Compute mean of translation
  Eigen::Vector3d mean_translation = Eigen::Vector3d::Zero();
  for (const auto &T : transforms) {
    mean_translation += T.translation();
  }
  mean_translation /= transforms.size();

  // Compute standard deviation of translation
  Eigen::Vector3d std_translation = Eigen::Vector3d::Zero();
  for (const auto &T : transforms) {
    Eigen::Vector3d diff = T.translation() - mean_translation;
    std_translation += diff.cwiseProduct(diff); // element-wise multiplication
  }
  std_translation = (std_translation / transforms.size())
                        .cwiseSqrt(); // element - wise square root

  Eigen::Matrix3d avg_rotation_tangent = Eigen::Matrix3d::Zero();
  Eigen::Vector3d avg_translation = Eigen::Vector3d::Zero();

  int valid_count = 0;
  for (const auto &T : transforms) {
    Eigen::Vector3d translation = T.translation();

    // Check if translation is within 3 sigma range
    if ((translation - mean_translation)
            .cwiseAbs()
            .cwiseQuotient(std_translation)
            .maxCoeff() <= 3.0) {
      avg_translation += translation;
      avg_rotation_tangent +=
          rotationMatrixToTangent(T.rotation());
      valid_count++;
    }
  }

  if (valid_count == 0) {
    // If all transforms are discarded, return identity.
    return Eigen::Isometry3d::Identity();
  }

  avg_rotation_tangent /= valid_count;
  avg_translation /= valid_count;

  Eigen::Matrix3d avg_rotation =
      tangentToRotationMatrix(avg_rotation_tangent);

  Eigen::Isometry3d avg_transform = Eigen::Isometry3d::Identity();
  avg_transform.linear() = avg_rotation;
  avg_transform.translation() = avg_translation;

  return avg_transform;
}

CloudPtr buildPointCloudMap(const std::vector<std::string> &cloud_paths,
                            const std::vector<Eigen::Isometry3d> &transforms,
                            const Eigen::Isometry3d T_ext) {
  CloudPtr map(new PointCloudType);
  for (size_t i = 0; i < cloud_paths.size(); ++i) {
    CloudPtr cloud(new PointCloudType);
    pcl::io::loadPCDFile(cloud_paths[i], *cloud);

    CloudPtr filtered_cloud(new PointCloudType);
    for (size_t j = 0; j < cloud->size(); ++j) {
      const auto &point = cloud->points[j];
      float distance =
          std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
      if (distance > 2 && distance < 150) {
        filtered_cloud->push_back(point);
      }
    }

    pcl::transformPointCloud(*filtered_cloud, *filtered_cloud,
                             (transforms[i] * T_ext).matrix());
    *map += *filtered_cloud;
  }

  pcl::VoxelGrid<PointType> vg;
  vg.setLeafSize(0.2, 0.2, 0.2);
  vg.setInputCloud(map);
  vg.filter(*map);

  pcl::StatisticalOutlierRemoval<PointType> sor;
  sor.setInputCloud(map);
  sor.setMeanK(30);
  sor.setStddevMulThresh(1.0);
  sor.filter(*map);

  return map;
}

double analyzeTerrainRoughness(const CloudPtr &cloud,
                              const std::vector<Eigen::Isometry3d> &poses) {
  double height = 0;

  for (size_t i = 0; i < poses.size(); ++i) {
    CloudPtr cloud_filtered(new PointCloudType);

    for (size_t j = 0; j < cloud->size(); ++j) {
      const auto &p = cloud->points[j];
      double distance = sqrt((p.x - poses[i](0, 3)) * (p.x - poses[i](0, 3)) +
                             (p.y - poses[i](1, 3)) * (p.y - poses[i](1, 3)) +
                             (p.z - poses[i](2, 3)) * (p.z - poses[i](2, 3)));
      if (distance > 2 && distance < 20)
        cloud_filtered->push_back(p);
    }

    pcl::ApproximateVoxelGrid<PointType> downsample;
    downsample.setLeafSize(0.1f, 0.1f, 0.1f);
    downsample.setInputCloud(cloud_filtered);
    downsample.filter(*cloud_filtered);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<PointType> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.05);

    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);

    pcl::PointCloud<PointType>::Ptr ground(new PointCloudType);
    pcl::ExtractIndices<PointType> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*ground);

    std::vector<float> distances;
    for (size_t j = 0; j < ground->size(); ++j) {
      Eigen::Vector4f ground_model;
      ground_model << coefficients->values[0], coefficients->values[1],
          coefficients->values[2], coefficients->values[3];
      distances.push_back(std::abs(ground_model.dot(
          Eigen::Vector4f((*ground)[j].x, (*ground)[j].y, (*ground)[j].z, 1))));
    }

    // Calculate mean distance
    float mean = std::accumulate(distances.begin(), distances.end(), 0.0) / distances.size();

    // Calculate standard deviation
    float variance = std::accumulate(distances.begin(), distances.end(), 0.0f,
                                     [&mean](float accumulator, float value) {
                                       return accumulator + (value - mean) * (value - mean);
                                     });
    float std_dev = std::sqrt(variance / distances.size());

    // Output ground roughness
    // std::cout << "Roughness: " << std_dev << std::endl;
  }

  return height;
}