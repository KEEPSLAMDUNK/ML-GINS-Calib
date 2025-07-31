// Extrinsic calibration optimization implementation for multi-LiDAR GNSS/INS system

#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "ml_gins_calib/optimization/extrinsic_calibrator.h"

calib::ExtrinsicCalibrator::ExtrinsicCalibrator() {}
calib::ExtrinsicCalibrator::~ExtrinsicCalibrator() {}

bool calib::ExtrinsicCalibrator::ComputeCovariances(
    const CloudPtr &cloud, nanoflann::KdTreeFLANN<PointType> &kdtree,
    std::shared_ptr<
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>>
        &covariances,
    int knn) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances->resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), knn, k_indices, k_sq_distances);

    Eigen::Matrix<double, 4, -1> neighbors(4, knn);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) =
          cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / knn;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d values = Eigen::Vector3d(1, 1, 1e-3);

    (*covariances)[i].setZero();
    (*covariances)[i].template block<3, 3>(0, 0) =
        svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
  }

  return true;
}

void calib::ExtrinsicCalibrator::AddInitValue(const uint64_t &key,
                                       const Eigen::Isometry3d &mat) {
  gtsam::Pose3 pose(mat.matrix());
  initial_estimate_.insert(gtsam::Key(key), pose);
}

void calib::ExtrinsicCalibrator::SolveGraphOptimization(
    std::vector<Eigen::Isometry3d> &res) {
  gtsam::LevenbergMarquardtParams optimizer_params;
  optimizer_params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  optimizer_params.setRelativeErrorTol(1e-4);

  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_,
                                               optimizer_params);
  gtsam::Values result = optimizer.optimize();

  res.resize(result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    gtsam::Pose3 T_final = result.at<gtsam::Pose3>(i);
    Eigen::Isometry3d T(T_final.matrix());
    res[i] = T;
  }
}

void calib::ExtrinsicCalibrator::ClearGraph() {
  graph_.resize(0);
  initial_estimate_.clear();
}

void calib::ExtrinsicCalibrator::AddGicpFactors(
    const uint64_t &key1, const uint64_t &key2,
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data) {
  std::vector<int> correspondences;
  Mat4dVec opt_cov;
  UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

  graph_.reserve(correspondences.size());

  for (size_t i = 0; i < correspondences.size(); ++i) {
    if (correspondences[i] < 0)
      continue;

    PointType point_a = corr_pre_data.source_cloud_->points[i];
    PointType point_b = corr_pre_data.target_cloud_->points[correspondences[i]];
    Eigen::Vector3d point_a_vec(point_a.x, point_a.y, point_a.z);
    Eigen::Vector3d point_b_vec(point_b.x, point_b.y, point_b.z);

    gtsam::noiseModel::Gaussian::shared_ptr gaussian_model =
        gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
    gtsam::noiseModel::Robust::shared_ptr noise_model =
        gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(0.1), gaussian_model);
    gtsam::MLFactor factor(gtsam::Key(key1), gtsam::Key(key2), point_a_vec,
                           point_b_vec, noise_model);

    graph_.push_back(factor);
  }
}

void calib::ExtrinsicCalibrator::UpdateCorrespondences(
    const CorrPreData &corr_pre_data, std::vector<int> &correspondences,
    Mat4dVec &opt_cov) {
  Eigen::Isometry3f source_trans_f = corr_pre_data.source_trans_.cast<float>();

  CloudPtr target_cloud(new PointCloudType);
  pcl::transformPointCloud(*corr_pre_data.target_cloud_, *target_cloud,
                           corr_pre_data.target_trans_.matrix());

  nanoflann::KdTreeFLANN<PointType>::Ptr target_kdtree(
      new nanoflann::KdTreeFLANN<PointType>);
  target_kdtree->setInputCloud(target_cloud);

  correspondences.resize(corr_pre_data.source_cloud_->size());
  opt_cov.resize(corr_pre_data.source_cloud_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_)                             \
    firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < corr_pre_data.source_cloud_->size(); i++) {
    if (sampling_) {
      if (i % int(1 / sampling_ratio_) != 0) {
        correspondences[i] = -1;
        continue;
      }
    }

    PointType pt;
    pt.getVector4fMap() =
        source_trans_f * corr_pre_data.source_cloud_->at(i).getVector4fMap();

    target_kdtree->nearestKSearch(pt, 1, k_indices, k_sq_dists);



    correspondences[i] = k_sq_dists[0] < corr_pre_data.corr_dist_threshold_ *
                                             corr_pre_data.corr_dist_threshold_
                             ? k_indices[0]
                             : -1;

    if (correspondences[i] < 0) {
      continue;
    }

    const int target_index = correspondences[i];
    const auto &cov_A = (*corr_pre_data.source_cov_)[i];
    const auto &cov_B = (*corr_pre_data.target_cov_)[target_index];

    opt_cov[i] = corr_pre_data.target_trans_ * cov_B *
                     corr_pre_data.target_trans_.matrix().transpose() +
                 corr_pre_data.source_trans_ * cov_A *
                     corr_pre_data.source_trans_.matrix().transpose();
  }
}

void calib::ExtrinsicCalibrator::AddPriorFactor(const uint64_t &key,
                                         const Eigen::Isometry3d &mat) {
  gtsam::SharedDiagonal noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
      gtsam::Key(key), gtsam::Pose3(mat.matrix()), noise));
}

Eigen::Isometry3d calib::ExtrinsicCalibrator::LgOptimizationCoarse(
    const std::vector<std::string> &paths,
    const std::vector<Eigen::Isometry3d> &poses,
    const Eigen::Isometry3d &T_ext_prior) {
  if (paths.empty() || (paths.size() != poses.size())) {
    std::cerr << "Empty paths!" << std::endl;
    return Eigen::Isometry3d::Identity();
  }

  auto load_cloud_func = [&](const std::string &path, CloudPtr &cloud,
                             float voxel_size = 0.2) {
    if (pcl::io::loadPCDFile(path, *cloud) == -1) {
      std::cerr << "Could not read point cloud data." << std::endl;
      return false;
    }

    CloudPtr filtered_cloud(new PointCloudType);
    for (int i = 0; i < cloud->size(); i++) {
      const PointType &p = cloud->points[i];
      float distance = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (distance > 2 && distance < 150) {
        filtered_cloud->push_back(p);
      }
    }

    pcl::ApproximateVoxelGrid<PointType> avg;
    avg.setLeafSize(voxel_size, voxel_size, voxel_size);
    avg.setInputCloud(filtered_cloud);
    avg.filter(*cloud);

    return true;
  };

  size_t size = paths.size();
  std::vector<CloudPtr> clouds(size);
  std::vector<std::shared_ptr<Mat4dVec>> cloud_covs(size);
  std::vector<nanoflann::KdTreeFLANN<PointType>::Ptr> cloud_kdtrees(size);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < paths.size(); ++i) {
    CloudPtr cloud(new pcl::PointCloud<PointType>);
    if (load_cloud_func(paths[i], cloud)) {
      clouds[i] = cloud;

      nanoflann::KdTreeFLANN<PointType>::Ptr cloud_kdtree(
          new nanoflann::KdTreeFLANN<PointType>);
      cloud_kdtree->setInputCloud(cloud);
      cloud_kdtrees[i] = cloud_kdtree;

      std::shared_ptr<Mat4dVec> cloud_cov(new Mat4dVec);
      this->ComputeCovariances(cloud, *cloud_kdtree, cloud_cov, 20);
      cloud_covs[i] = cloud_cov;
    }
  }

  if (clouds.size() < 2) {
    std::cerr << "At least 2 valid point clouds are required." << std::endl;
    return Eigen::Isometry3d::Identity();
  }

  // Calculate ground-to-lidar transformation
  Eigen::Vector4d ground_model;
  GroundExtraction(clouds[0], ground_model);
  Eigen::Vector3d vec_before, vec_after;
  vec_before << ground_model[0], ground_model[1], ground_model[2];
  vec_after << 0, 0, 1;

  Eigen::Matrix3d rot_mat =
      Eigen::Quaterniond::FromTwoVectors(vec_before, vec_after)
          .toRotationMatrix();

  Eigen::Isometry3d T_ground_lidar = Eigen::Isometry3d::Identity();
  T_ground_lidar.linear() = rot_mat;
  T_ground_lidar.translation().z() = ground_model[3];
  T_ground_lidar_ = T_ground_lidar;

  std::cout << "height: " << ground_model[3] << std::endl;

  std::vector<ExtrinsicCalibrator::CorrPreData> corr_pre_datas;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      ExtrinsicCalibrator::CorrPreData corr_pre_data;
      corr_pre_data.source_cloud_ = clouds[i];
      corr_pre_data.target_cloud_ = clouds[j];
      corr_pre_data.source_cov_ = cloud_covs[i];
      corr_pre_data.target_cov_ = cloud_covs[j];
      corr_pre_data.source_trans_ = poses[i];
      corr_pre_data.target_trans_ = poses[j];
      corr_pre_data.corr_dist_threshold_ = 1.0;
      corr_pre_datas.emplace_back(corr_pre_data);
    }
  }

  // Transform prior extrinsic from lidar frame to ground frame
  Eigen::Isometry3d T_ext(T_ext_prior * T_ground_lidar.inverse());
  for (size_t iter = 0; iter < opt_iter_; ++iter) {
    for (size_t i = 0; i < corr_pre_datas.size(); ++i) {
      this->AddLgGicpFactors(0, corr_pre_datas[i], T_ext * T_ground_lidar);
    }

    this->AddInitValue(0, T_ext);
    if (ltg_prior_factor_)
      this->AddLgPriorFactor(0, T_ext);

    this->AddPlaneLgPriorFactor(0, T_ground_imu_);

    std::vector<Eigen::Isometry3d> T_vec;
    this->SolveGraphOptimization(T_vec);
    this->ClearGraph();

    auto last_T_ext = T_ext;
    T_ext = T_vec[0];

    T_exts_.push_back(T_ext);

    if (IsConverged(last_T_ext, T_ext)) {
      std::cout << "Iteration: " << iter << std::endl;
      break;
    }
  }

  auto T_res = T_ext * T_ground_lidar;
  std::cout << "Extrinsic:" << std::endl;
  std::cout << T_res.matrix() << std::endl << std::endl;

  return T_res;
}

void calib::ExtrinsicCalibrator::UpdateCorrespondences(
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data,
    const Eigen::Isometry3d &T_ext, std::vector<int> &correspondences,
    Mat4dVec &opt_cov) {
  Eigen::Isometry3f T_ext_f = T_ext.cast<float>();
  Eigen::Isometry3f source_trans_f = corr_pre_data.source_trans_.cast<float>();

  CloudPtr target_cloud(new PointCloudType);
  pcl::transformPointCloud(*corr_pre_data.target_cloud_, *target_cloud,
                           (corr_pre_data.target_trans_ * T_ext).matrix());

  nanoflann::KdTreeFLANN<PointType>::Ptr target_kdtree(
      new nanoflann::KdTreeFLANN<PointType>);
  target_kdtree->setInputCloud(target_cloud);

  correspondences.resize(corr_pre_data.source_cloud_->size());
  opt_cov.resize(corr_pre_data.source_cloud_->size());

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);

#pragma omp parallel for num_threads(num_threads_)                             \
    firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < corr_pre_data.source_cloud_->size(); i++) {
    if (i % 100 != 0) {
      correspondences[i] = -1;
      continue;
    }

    PointType pt;
    pt.getVector4fMap() = source_trans_f * T_ext_f *
                          corr_pre_data.source_cloud_->at(i).getVector4fMap();

    target_kdtree->nearestKSearch(pt, 1, k_indices, k_sq_dists);

    correspondences[i] = k_sq_dists[0] < corr_pre_data.corr_dist_threshold_ *
                                             corr_pre_data.corr_dist_threshold_
                             ? k_indices[0]
                             : -1;

    if (correspondences[i] < 0) {
      continue;
    }

    const int target_index = correspondences[i];
    const auto &cov_A = (*corr_pre_data.source_cov_)[i];
    const auto &cov_B = (*corr_pre_data.target_cov_)[target_index];

    opt_cov[i] =
        corr_pre_data.target_trans_ * T_ext * cov_B *
            (corr_pre_data.target_trans_ * T_ext).matrix().transpose() +
        corr_pre_data.source_trans_ * T_ext * cov_A *
            (corr_pre_data.source_trans_ * T_ext).matrix().transpose();
  }
}

void calib::ExtrinsicCalibrator::AddLgGicpFactors(const uint64_t &key,
                                           const CorrPreData &corr_pre_data,
                                           const Eigen::Isometry3d &T_ext) {
  std::vector<int> correspondences;
  Mat4dVec opt_cov;
  UpdateCorrespondences(corr_pre_data, T_ext, correspondences, opt_cov);

  graph_.reserve(correspondences.size());

  for (size_t i = 0; i < correspondences.size(); ++i) {
    if (correspondences[i] < 0)
      continue;

    PointType point_a = corr_pre_data.source_cloud_->points[i];
    PointType point_b = corr_pre_data.target_cloud_->points[correspondences[i]];
    Eigen::Vector3d point_a_vec(point_a.x, point_a.y, point_a.z);
    Eigen::Vector3d point_b_vec(point_b.x, point_b.y, point_b.z);
    point_a_vec = T_ground_lidar_ * point_a_vec;
    point_b_vec = T_ground_lidar_ * point_b_vec;
    gtsam::Pose3 pose_a(corr_pre_data.source_trans_.matrix());
    gtsam::Pose3 pose_b(corr_pre_data.target_trans_.matrix());

    gtsam::noiseModel::Gaussian::shared_ptr gaussian_model =
        gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
    gtsam::noiseModel::Robust::shared_ptr noise_model =
        gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(0.1), gaussian_model);
    gtsam::LgFactor factor(gtsam::Key(key), point_a_vec, point_b_vec, pose_a,
                           pose_b, noise_model);

    graph_.push_back(factor);
  }
}

void calib::ExtrinsicCalibrator::AddLgPriorFactor(const uint64_t &key,
                                           const Eigen::Isometry3d &T_ext) {
  double angle_std = 0.01;
  double trans_std = 1e8;

  gtsam::SharedDiagonal noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << angle_std, angle_std, angle_std, trans_std,
       trans_std, trans_std)
          .finished());
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(
      gtsam::Key(key), gtsam::Pose3(T_ext.matrix()), noise));
}

void calib::ExtrinsicCalibrator::GroundExtraction(const CloudPtr &cloud,
                                           Eigen::Vector4d &ground_model) {
  CloudPtr cloud_filtered(new PointCloudType);


  *cloud_filtered = *cloud;

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
  seg.setAxis(Eigen::Vector3f(0, 0, 1));
  seg.setEpsAngle(M_PI_4);
  seg.setDistanceThreshold(0.1);

  seg.setInputCloud(cloud_filtered);
  seg.segment(*inliers, *coefficients);

  pcl::PointCloud<PointType>::Ptr ground(new PointCloudType);
  pcl::ExtractIndices<PointType> extract;
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*ground);

  pcl::PointCloud<PointType>::Ptr non_ground(new PointCloudType);
  extract.setInputCloud(cloud_filtered);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*non_ground);

  Eigen::Vector4f ground_centroid;
  pcl::compute3DCentroid(*ground, ground_centroid);
  Eigen::Vector4f non_ground_centroid;
  pcl::compute3DCentroid(*non_ground, non_ground_centroid);

  Eigen::Vector3d v;
  v[0] = non_ground_centroid[0] - ground_centroid[0];
  v[1] = non_ground_centroid[1] - ground_centroid[1];
  v[2] = non_ground_centroid[2] - ground_centroid[2];
  float dot_product = coefficients->values[0] * v[0] +
                      coefficients->values[1] * v[1] +
                      coefficients->values[2] * v[2];

  if (dot_product > 0)
    ground_model << coefficients->values[0], coefficients->values[1],
        coefficients->values[2], coefficients->values[3];
  else
    ground_model << -coefficients->values[0], -coefficients->values[1],
        -coefficients->values[2], -coefficients->values[3];
}

Eigen::Isometry3d
calib::ExtrinsicCalibrator::GroundAlignment(const CloudPtr &source_cloud,
                                     const CloudPtr &target_cloud) {
  Eigen::Vector4d source_ground_model;
  GroundExtraction(source_cloud, source_ground_model);
  Eigen::Vector4d target_ground_model;
  GroundExtraction(target_cloud, target_ground_model);

  Eigen::Vector3d vec_before, vec_after;
  vec_before << source_ground_model[0], source_ground_model[1],
      source_ground_model[2];
  vec_after << target_ground_model[0], target_ground_model[1],
      target_ground_model[2];

  Eigen::Matrix3d rot_mat =
      Eigen::Quaterniond::FromTwoVectors(vec_before, vec_after)
          .toRotationMatrix();

  Eigen::Isometry3d T_target_source = Eigen::Isometry3d::Identity();
  T_target_source.linear() = rot_mat;
  T_target_source.translation().z() =
      source_ground_model[3] - target_ground_model[3];

  return T_target_source;
}

void calib::ExtrinsicCalibrator::AddLgGicpFactors(const uint64_t &key1,
                                           const uint64_t &key2,
                                           const uint64_t &key3,
                                           const CorrPreData &corr_pre_data,
                                           const Eigen::Isometry3d &T_ext) {
  std::vector<int> correspondences;
  Mat4dVec opt_cov;
  UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

  graph_.reserve(correspondences.size());

  for (size_t i = 0; i < correspondences.size(); ++i) {
    if (correspondences[i] < 0)
      continue;

    PointType point_a = corr_pre_data.source_cloud_->points[i];
    PointType point_b = corr_pre_data.target_cloud_->points[correspondences[i]];
    Eigen::Vector3d point_a_vec(point_a.x, point_a.y, point_a.z);
    Eigen::Vector3d point_b_vec(point_b.x, point_b.y, point_b.z);
    point_a_vec = T_ground_lidar_ * point_a_vec;
    point_b_vec = T_ground_lidar_ * point_b_vec;

    gtsam::noiseModel::Gaussian::shared_ptr gaussian_model =
        gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
    gtsam::noiseModel::Robust::shared_ptr noise_model =
        gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(0.1), gaussian_model);
    gtsam::LgFactor2 factor(gtsam::Key(key1), gtsam::Key(key2),
                            gtsam::Key(key3), point_a_vec, point_b_vec,
                            noise_model);

    graph_.push_back(factor);
  }
}

Eigen::Isometry3d calib::ExtrinsicCalibrator::LgOptimizationRefine(
    const std::vector<std::string> &paths,
    std::vector<Eigen::Isometry3d> &poses,
    const Eigen::Isometry3d &T_ext_prior) {
  if (paths.empty() || (paths.size() != poses.size())) {
    std::cerr << "Empty paths!" << std::endl;
    return Eigen::Isometry3d::Identity();
  }

  auto load_cloud_func = [&](const std::string &path, CloudPtr &cloud,
                             float voxel_size = 0.2) {
    if (pcl::io::loadPCDFile(path, *cloud) == -1) {
      std::cerr << "Could not read point cloud data." << std::endl;
      return false;
    }

    CloudPtr filtered_cloud(new PointCloudType);
    for (int i = 0; i < cloud->size(); i++) {
      const PointType &p = cloud->points[i];
      float distance = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (distance > 2 && distance < 150) {
        filtered_cloud->push_back(p);
      }
    }

    pcl::ApproximateVoxelGrid<PointType> avg;
    avg.setLeafSize(voxel_size, voxel_size, voxel_size);
    avg.setInputCloud(filtered_cloud);
    avg.filter(*cloud);

    return true;
  };

  size_t size = paths.size();
  std::vector<CloudPtr> clouds(size);
  std::vector<std::shared_ptr<Mat4dVec>> cloud_covs(size);
  std::vector<nanoflann::KdTreeFLANN<PointType>::Ptr> cloud_kdtrees(size);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < paths.size(); ++i) {
    CloudPtr cloud(new pcl::PointCloud<PointType>);
    if (load_cloud_func(paths[i], cloud)) {
      clouds[i] = cloud;

      nanoflann::KdTreeFLANN<PointType>::Ptr cloud_kdtree(
          new nanoflann::KdTreeFLANN<PointType>);
      cloud_kdtree->setInputCloud(cloud);
      cloud_kdtrees[i] = cloud_kdtree;

      std::shared_ptr<Mat4dVec> cloud_cov(new Mat4dVec);
      this->ComputeCovariances(cloud, *cloud_kdtree, cloud_cov, 20);
      cloud_covs[i] = cloud_cov;
    }
  }

  if (clouds.size() < 2) {
    std::cerr << "At least 2 valid point clouds are required." << std::endl;
    return Eigen::Isometry3d::Identity();
  }

  std::vector<std::pair<int, int>> index_pairs;
  std::vector<ExtrinsicCalibrator::CorrPreData> corr_pre_datas;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      index_pairs.emplace_back(i, j);

      ExtrinsicCalibrator::CorrPreData corr_pre_data;
      corr_pre_data.source_cloud_ = clouds[i];
      corr_pre_data.target_cloud_ = clouds[j];
      corr_pre_data.source_cov_ = cloud_covs[i];
      corr_pre_data.target_cov_ = cloud_covs[j];
      corr_pre_data.source_trans_ = poses[i] * T_ext_prior;
      corr_pre_data.target_trans_ = poses[j] * T_ext_prior;
      corr_pre_data.corr_dist_threshold_ = 1.0;
      corr_pre_datas.emplace_back(corr_pre_data);
    }
  }

  Eigen::Isometry3d T_ext(T_ext_prior * T_ground_lidar_.inverse());
  for (size_t iter = 0; iter < opt_iter_; ++iter) {
    for (size_t i = 0; i < corr_pre_datas.size(); ++i) {
      int key1 = index_pairs[i].first + 1;
      int key2 = index_pairs[i].second + 1;

      this->AddLgGicpFactors(key1, 0, key2, corr_pre_datas[i], T_ext * T_ground_lidar_);
    }

    this->AddInitValue(0, T_ext);
    this->AddLgPriorFactor2(0, T_ext);
    for (size_t i = 0; i < poses.size(); ++i) {
      this->AddInitValue(i + 1, poses[i]);
      this->AddLgPriorFactor2(i + 1, poses[i]);
    }

    this->AddPlaneLgPriorFactor(0, T_ground_imu_);

    std::vector<Eigen::Isometry3d> T_vec;
    this->SolveGraphOptimization(T_vec);
    this->ClearGraph();

    std::vector<Eigen::Isometry3d> last_estimated_poses;
    last_estimated_poses.push_back(T_ext);
    last_estimated_poses.insert(last_estimated_poses.end(), poses.begin(),
                                poses.end());

    T_ext = T_vec[0];
    for (size_t i = 1; i < T_vec.size(); ++i) {
      poses[i - 1] = T_vec[i];
    }

    for (size_t i = 0; i < corr_pre_datas.size(); ++i) {
      int key1 = index_pairs[i].first;
      int key2 = index_pairs[i].second;

      corr_pre_datas[i].source_trans_ = poses[key1] * T_ext * T_ground_lidar_;
      corr_pre_datas[i].target_trans_ = poses[key2] * T_ext * T_ground_lidar_;
    }

    T_exts_.emplace_back(T_ext);

    if (IsConverged(last_estimated_poses, T_vec)) {
      std::cout << "Iteration: " << iter << std::endl;
      break;
    }
  }

  auto T_res = T_ext * T_ground_lidar_;
  std::cout << "Extrinsic:" << std::endl;
  std::cout << T_res.matrix() << std::endl << std::endl;

  return T_res;
}

void calib::ExtrinsicCalibrator::AddLgPriorFactor2(const uint64_t &key,
                                            const Eigen::Isometry3d &T) {
  double angle_std = 0.01;
  double trans_std = 0.01;

  gtsam::SharedDiagonal noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << angle_std, angle_std, angle_std, trans_std,
       trans_std, trans_std)
          .finished());
  graph_.add(gtsam::PriorFactor<gtsam::Pose3>(gtsam::Key(key),
                                              gtsam::Pose3(T.matrix()), noise));
}
void calib::ExtrinsicCalibrator::JointOptimization(
    const std::vector<std::vector<std::string>> &cloud_paths,
    std::vector<Eigen::Isometry3d> &gins_poses,
    std::vector<Eigen::Isometry3d> &ltl_ext_prior,
    const Eigen::Isometry3d &ltg_ext_prior) {
  if (cloud_paths.size() != ltl_ext_prior.size() ||
      cloud_paths[0].size() != gins_poses.size()) {
    std::cerr << "Invalid input data." << std::endl;
  }

  auto load_cloud_func = [&](const std::string &path, CloudPtr &cloud,
                             float voxel_size = 0.2) {
    if (pcl::io::loadPCDFile(path, *cloud) == -1) {
      std::cerr << "Could not read point cloud data." << std::endl;
      return false;
    }

    CloudPtr filtered_cloud(new PointCloudType);
    for (int i = 0; i < cloud->size(); i++) {
      const PointType &p = cloud->points[i];
      float distance = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      if (distance > 2 && distance < 150) {
        filtered_cloud->push_back(p);
      }
    }

    pcl::ApproximateVoxelGrid<PointType> avg;
    avg.setLeafSize(voxel_size, voxel_size, voxel_size);
    avg.setInputCloud(filtered_cloud);
    avg.filter(*cloud);



    return true;
  };

  size_t row_size = cloud_paths.size();
  size_t col_size = cloud_paths[0].size();
  std::vector<std::vector<CloudPtr>> clouds(row_size,
                                            std::vector<CloudPtr>(col_size));
  std::vector<std::vector<std::shared_ptr<Mat4dVec>>> cloud_covs(
      row_size, std::vector<std::shared_ptr<Mat4dVec>>(col_size));
  std::vector<std::vector<nanoflann::KdTreeFLANN<PointType>::Ptr>>
      cloud_kdtrees(
          row_size,
          std::vector<nanoflann::KdTreeFLANN<PointType>::Ptr>(col_size));
  std::vector<std::vector<std::vector<float>>> planar_metrics(
      row_size, std::vector<std::vector<float>>(col_size));
  std::vector<std::vector<std::shared_ptr<GaussianVoxelMap<PointType>>>>
      voxel_maps(
          row_size,
          std::vector<std::shared_ptr<GaussianVoxelMap<PointType>>>(col_size));

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < row_size; ++i) {
    for (int j = 0; j < col_size; ++j) {
      CloudPtr cloud(new pcl::PointCloud<PointType>);
      std::shared_ptr<GaussianVoxelMap<PointType>> voxel_map;
      voxel_map.reset(
          new GaussianVoxelMap<PointType>(voxel_resolution_, voxel_mode_));

      if (load_cloud_func(cloud_paths[i][j], cloud)) {
        clouds[i][j] = cloud;

        nanoflann::KdTreeFLANN<PointType>::Ptr cloud_kdtree(
            new nanoflann::KdTreeFLANN<PointType>);
        cloud_kdtree->setInputCloud(cloud);
        cloud_kdtrees[i][j] = cloud_kdtree;

        std::shared_ptr<Mat4dVec> cloud_cov(new Mat4dVec);
        std::vector<float> planarity;
        //        this->ComputeCovariances(cloud, *cloud_kdtree, cloud_cov, 20);
        this->ComputeCovariancesAndPlanarity(cloud, *cloud_kdtree, cloud_cov,
                                             planarity);
        cloud_covs[i][j] = cloud_cov;
        planar_metrics[i][j] = planarity;
        voxel_maps[i][j] = voxel_map;
      }
    }
  }

  std::vector<Eigen::Isometry3d> lidar_poses(col_size);
  for (size_t i = 0; i < col_size; ++i) {
    lidar_poses[i] = gins_poses[i] * ltg_ext_prior;
  }
  auto first_lidar_pose = lidar_poses[0];
  for (size_t i = 0; i < col_size; ++i) {
    lidar_poses[i] = first_lidar_pose.inverse() * lidar_poses[i];
  }

  std::vector<Eigen::Isometry3d> ltl_exts = ltl_ext_prior;

  std::vector<Eigen::Isometry3d> gins_poses_estimated = gins_poses;
  Eigen::Isometry3d ltg_ext_estimated = ltg_ext_prior;

  // one extrinsic, two poses
  std::vector<ExtrinsicCalibrator::CorrPreData> corr_pre_datas1;
  for (size_t i = 0; i < row_size; ++i) {
    for (size_t j = 0; j < col_size; ++j) {
      for (size_t k = j + 1; k < col_size; ++k) {
        ExtrinsicCalibrator::CorrPreData corr_pre_data;
        corr_pre_data.source_cloud_ = clouds[i][j];
        corr_pre_data.target_cloud_ = clouds[i][k];
        corr_pre_data.source_cov_ = cloud_covs[i][j];
        corr_pre_data.target_cov_ = cloud_covs[i][k];
        corr_pre_data.source_trans_ = lidar_poses[j] * ltl_exts[i];
        corr_pre_data.target_trans_ = lidar_poses[k] * ltl_exts[i];
        corr_pre_data.source_planarity_ = planar_metrics[i][j];
        corr_pre_data.target_planarity_ = planar_metrics[i][k];
        corr_pre_data.voxel_map_ = voxel_maps[i][k];
        corr_pre_data.corr_dist_threshold_ = max_corr_dis_;
        corr_pre_datas1.emplace_back(corr_pre_data);
      }
    }
  }

  // two extrinsics, one poses
  std::vector<ExtrinsicCalibrator::CorrPreData> corr_pre_datas2;
  for (size_t i = 0; i < col_size; ++i) {
    for (size_t j = 0; j < row_size; ++j) {
      for (size_t k = j + 1; k < row_size; ++k) {
        ExtrinsicCalibrator::CorrPreData corr_pre_data;
        corr_pre_data.source_cloud_ = clouds[j][i];
        corr_pre_data.target_cloud_ = clouds[k][i];
        corr_pre_data.source_cov_ = cloud_covs[j][i];
        corr_pre_data.target_cov_ = cloud_covs[k][i];
        corr_pre_data.source_trans_ = lidar_poses[i] * ltl_exts[j];
        corr_pre_data.target_trans_ = lidar_poses[i] * ltl_exts[k];
        corr_pre_data.source_planarity_ = planar_metrics[j][i];
        corr_pre_data.target_planarity_ = planar_metrics[k][i];
        corr_pre_data.voxel_map_ = voxel_maps[k][i];
        corr_pre_data.corr_dist_threshold_ = max_corr_dis_;
        corr_pre_datas2.emplace_back(corr_pre_data);
      }
    }
  }

  // two extrinsics, two poses
  std::vector<ExtrinsicCalibrator::CorrPreData> corr_pre_datas3;
  for (size_t i = 0; i < row_size; ++i) {
    for (size_t j = 0; j < col_size; ++j) {
      for (size_t k = i + 1; k < row_size; ++k) {
        for (size_t l = j + 1; l < col_size; ++l) {
          ExtrinsicCalibrator::CorrPreData corr_pre_data;
          corr_pre_data.source_cloud_ = clouds[i][j];
          corr_pre_data.target_cloud_ = clouds[k][l];
          corr_pre_data.source_cov_ = cloud_covs[i][j];
          corr_pre_data.target_cov_ = cloud_covs[k][l];
          corr_pre_data.source_trans_ = lidar_poses[j] * ltl_exts[i];
          corr_pre_data.target_trans_ = lidar_poses[l] * ltl_exts[k];
          corr_pre_data.source_planarity_ = planar_metrics[i][j];
          corr_pre_data.target_planarity_ = planar_metrics[k][l];
          corr_pre_data.voxel_map_ = voxel_maps[k][l];
          corr_pre_data.corr_dist_threshold_ = max_corr_dis_;
          corr_pre_datas3.emplace_back(corr_pre_data);
        }
      }
    }
  }

  std::vector<ExtrinsicCalibrator::CorrPreData> corr_pre_datas4;
  for (size_t i = 0; i < col_size; ++i) {
    for (size_t j = i + 1; j < col_size; ++j) {
      ExtrinsicCalibrator::CorrPreData corr_pre_data;
      corr_pre_data.source_cloud_ = clouds[0][i];
      corr_pre_data.target_cloud_ = clouds[0][j];
      corr_pre_data.source_cov_ = cloud_covs[0][i];
      corr_pre_data.target_cov_ = cloud_covs[0][j];
      corr_pre_data.source_trans_ = gins_poses[i] * ltg_ext_prior;
      corr_pre_data.target_trans_ = gins_poses[j] * ltg_ext_prior;
      corr_pre_data.source_planarity_ = planar_metrics[0][i];
      corr_pre_data.target_planarity_ = planar_metrics[0][j];
      corr_pre_data.voxel_map_ = voxel_maps[0][j];
      corr_pre_data.corr_dist_threshold_ = max_corr_dis_;
      corr_pre_datas4.emplace_back(corr_pre_data);
    }
  }

  for (size_t iter = 0; iter < opt_iter_; ++iter) {
    current_opt_iter_ = iter;

    for (size_t i = 0; i < col_size; ++i) {
      this->AddInitValue(gtsam::Symbol('x', i), lidar_poses[i]);
    }

    for (size_t i = 1; i < row_size; ++i) {
      this->AddInitValue(gtsam::Symbol('e', i), ltl_exts[i]);
    }



    size_t count = 0;
    for (size_t i = 0; i < row_size; ++i) {
      for (size_t j = 0; j < col_size; ++j) {
        for (size_t k = j + 1; k < col_size; ++k) {
          if (i == 0)
            this->AddLtlFactors33(gtsam::Symbol('x', j), gtsam::Symbol('x', k),
                                  corr_pre_datas1[count]);
          else
            this->AddLtlFactors3(gtsam::Symbol('x', j), gtsam::Symbol('e', i),
                                 gtsam::Symbol('x', k), corr_pre_datas1[count]);
          ++count;
        }
      }
    }

    count = 0;
    for (size_t i = 0; i < col_size; ++i) {
      for (size_t j = 0; j < row_size; ++j) {
        for (size_t k = j + 1; k < row_size; ++k) {
          if (j == 0)
            this->AddLtlFactors22(gtsam::Symbol('x', i), gtsam::Symbol('e', k),
                                  corr_pre_datas2[count]);
          else
            this->AddLtlFactors2(gtsam::Symbol('x', i), gtsam::Symbol('e', j),
                                 gtsam::Symbol('e', k), corr_pre_datas2[count]);
          ++count;
        }
      }
    }

    count = 0;
    for (size_t i = 0; i < row_size; ++i) {
      for (size_t j = 0; j < col_size; ++j) {
        for (size_t k = i + 1; k < row_size; ++k) {
          for (size_t l = j + 1; l < col_size; ++l) {
            if (i == 0)
              this->AddLtlFactors11(
                  gtsam::Symbol('x', j), gtsam::Symbol('x', l),
                  gtsam::Symbol('e', k), corr_pre_datas3[count]);
            else
              this->AddLtlFactors1(gtsam::Symbol('x', j), gtsam::Symbol('e', i),
                                   gtsam::Symbol('x', l), gtsam::Symbol('e', k),
                                   corr_pre_datas3[count]);
            ++count;
          }
        }
      }
    }



    this->AddPriorFactor(gtsam::Symbol('x', 0), Eigen::Isometry3d::Identity());



    std::cout << graph_.size() << std::endl;

    auto result = this->SolveGraphOptimization();
    this->ClearGraph();

    auto last_lidar_poses = lidar_poses;
    auto last_ltl_exts = ltl_exts;
    auto last_gins_poses = gins_poses_estimated;
    auto last_ltg_ext = ltg_ext_estimated;

    for (size_t i = 0; i < col_size; ++i)
      lidar_poses[i] = result.at<gtsam::Pose3>(gtsam::Symbol('x', i)).matrix();
    for (size_t i = 1; i < row_size; ++i)
      ltl_exts[i] = result.at<gtsam::Pose3>(gtsam::Symbol('e', i)).matrix();


    if (IsConverged(last_lidar_poses, lidar_poses) &&
        IsConverged(last_ltl_exts, ltl_exts) &&
        IsConverged(last_gins_poses, gins_poses_estimated) &&
        IsConverged(last_ltg_ext, ltg_ext_estimated)) {
      std::cout << "Iteration: " << iter << std::endl;
      break;
    }

    count = 0;
    for (size_t i = 0; i < row_size; ++i) {
      for (size_t j = 0; j < col_size; ++j) {
        for (size_t k = j + 1; k < col_size; ++k) {
          corr_pre_datas1[count].source_trans_ = lidar_poses[j] * ltl_exts[i];
          corr_pre_datas1[count].target_trans_ = lidar_poses[k] * ltl_exts[i];
          corr_pre_datas1[count].corr_dist_threshold_ = max_corr_dis_;

          ++count;
        }
      }
    }

    count = 0;
    for (size_t i = 0; i < col_size; ++i) {
      for (size_t j = 0; j < row_size; ++j) {
        for (size_t k = j + 1; k < row_size; ++k) {
          corr_pre_datas2[count].source_trans_ = lidar_poses[i] * ltl_exts[j];
          corr_pre_datas2[count].target_trans_ = lidar_poses[i] * ltl_exts[k];
          corr_pre_datas2[count].corr_dist_threshold_ = max_corr_dis_;

          ++count;
        }
      }
    }

    count = 0;
    for (size_t i = 0; i < row_size; ++i) {
      for (size_t j = 0; j < col_size; ++j) {
        for (size_t k = i + 1; k < row_size; ++k) {
          for (size_t l = j + 1; l < col_size; ++l) {
            corr_pre_datas3[count].source_trans_ = lidar_poses[j] * ltl_exts[i];
            corr_pre_datas3[count].target_trans_ = lidar_poses[l] * ltl_exts[k];
            corr_pre_datas3[count].corr_dist_threshold_ = max_corr_dis_;

            ++count;
          }
        }
      }
    }

    count = 0;
    for (size_t i = 0; i < col_size; ++i) {
      for (size_t j = i + 1; j < col_size; ++j) {
        corr_pre_datas4[count].source_trans_ =
            gins_poses_estimated[i] * ltg_ext_estimated;
        corr_pre_datas4[count].target_trans_ =
            gins_poses_estimated[j] * ltg_ext_estimated;
        ++count;
      }
    }
  }

  first_lidar_pose = lidar_poses[0];
  for (size_t i = 0; i < col_size; ++i) {
    lidar_poses[i] = first_lidar_pose.inverse() * lidar_poses[i];
  }
  auto first_ltl_ext = ltl_exts[0];
  for (size_t i = 0; i < row_size; ++i) {
    ltl_exts[i] = first_ltl_ext.inverse() * ltl_exts[i];
  }

  std::cout << "Ltg: " << std::endl;
  std::cout << ltg_ext_prior.matrix() << std::endl << std::endl;
  std::cout << ltg_ext_estimated.matrix() << std::endl << std::endl;

  gins_poses = lidar_poses;
  ltl_ext_prior = ltl_exts;
}

// two poses, two extrinsics
void calib::ExtrinsicCalibrator::AddLtlFactors1(const gtsam::Symbol &lidar1,
                                         const gtsam::Symbol &lidar2,
                                         const gtsam::Symbol &lidar3,
                                         const gtsam::Symbol &lidar4,
                                         const CorrPreData &corr_pre_data) {
  if (huber_enable_) {
    std::vector<int> correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

    for (size_t i = 0; i < correspondences.size(); ++i) {
      if (correspondences[i] < 0) {
        continue;
      }

      PointType point_a = corr_pre_data.source_cloud_->points[i];
      PointType point_b =
          corr_pre_data.target_cloud_->points[correspondences[i]];
      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b.x, point_b.y, point_b.z);

      if (current_opt_iter_ < robust_opt_iter_) {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::noiseModel::Robust::shared_ptr robust_model =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1),
                gausian_model);
        gtsam::LtLFactor1 factor(lidar1, lidar2, lidar3, lidar4, pt_a, pt_b,
                                 robust_model);

        graph_.add(factor);
      } else {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::LtLFactor1 factor(lidar1, lidar2, lidar3, lidar4, pt_a, pt_b,
                                 gausian_model);

        graph_.add(factor);
      }
    }
  } else {
    std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, voxel_correspondences, opt_cov);

    for (size_t i = 0; i < voxel_correspondences.size(); ++i) {
      const auto &corr = voxel_correspondences[i];

      PointType point_a = corr_pre_data.source_cloud_->points[corr.first];
      Eigen::Vector4d point_b =
          corr_pre_data.target_trans_.inverse() * corr.second->mean;

      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b[0], point_b[1], point_b[2]);

      gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
          gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
      gtsam::noiseModel::Robust::shared_ptr robust_model =
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(voxel_resolution_ /
                                                           2.0),
              gausian_model);
      gtsam::LtLFactor1 factor(lidar1, lidar2, lidar3, lidar4, pt_a, pt_b,
                               robust_model, 1.0);

      graph_.add(factor);
    }
  }
}

// one pose, two extrinsics
void calib::ExtrinsicCalibrator::AddLtlFactors2(const gtsam::Symbol &lidar1,
                                         const gtsam::Symbol &lidar2,
                                         const gtsam::Symbol &lidar3,
                                         const CorrPreData &corr_pre_data) {
  if (huber_enable_) {
    std::vector<int> correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

    for (size_t i = 0; i < correspondences.size(); ++i) {
      if (correspondences[i] < 0) {
        continue;
      }

      PointType point_a = corr_pre_data.source_cloud_->points[i];
      PointType point_b =
          corr_pre_data.target_cloud_->points[correspondences[i]];
      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b.x, point_b.y, point_b.z);

      if (current_opt_iter_ < robust_opt_iter_) {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::noiseModel::Robust::shared_ptr robust_model =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1),
                gausian_model);
        gtsam::LtLFactor2 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                 robust_model);

        graph_.add(factor);
      } else {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::LtLFactor2 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                 gausian_model);

        graph_.add(factor);
      }
    }
  } else {
    std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, voxel_correspondences, opt_cov);

    for (size_t i = 0; i < voxel_correspondences.size(); ++i) {
      const auto &corr = voxel_correspondences[i];

      PointType point_a = corr_pre_data.source_cloud_->points[corr.first];
      Eigen::Vector4d point_b =
          corr_pre_data.target_trans_.inverse() * corr.second->mean;

      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b[0], point_b[1], point_b[2]);

      gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
          gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
      gtsam::noiseModel::Robust::shared_ptr robust_model =
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(voxel_resolution_ /
                                                           2.0),
              gausian_model);
      gtsam::LtLFactor2 factor(lidar1, lidar2, lidar3, pt_a, pt_b, robust_model,
                               1.0);

      graph_.add(factor);
    }
  }
}

// one extrinsic, two poses
void calib::ExtrinsicCalibrator::AddLtlFactors3(const gtsam::Symbol &lidar1,
                                         const gtsam::Symbol &lidar2,
                                         const gtsam::Symbol &lidar3,
                                         const CorrPreData &corr_pre_data) {
  if (huber_enable_) {
    std::vector<int> correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

    for (size_t i = 0; i < correspondences.size(); ++i) {
      if (correspondences[i] < 0) {
        continue;
      }

      PointType point_a = corr_pre_data.source_cloud_->points[i];
      PointType point_b =
          corr_pre_data.target_cloud_->points[correspondences[i]];
      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b.x, point_b.y, point_b.z);

      if (current_opt_iter_ < robust_opt_iter_) {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::noiseModel::Robust::shared_ptr robust_model =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1),
                gausian_model);
        gtsam::LtLFactor3 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                 robust_model);

        graph_.add(factor);
      } else {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::LtLFactor3 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                 gausian_model);

        graph_.add(factor);
      }
    }
  } else {
    std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, voxel_correspondences, opt_cov);

    for (size_t i = 0; i < voxel_correspondences.size(); ++i) {
      const auto &corr = voxel_correspondences[i];

      PointType point_a = corr_pre_data.source_cloud_->points[corr.first];
      Eigen::Vector4d point_b =
          corr_pre_data.target_trans_.inverse() * corr.second->mean;

      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b[0], point_b[1], point_b[2]);

      gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
          gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
      gtsam::noiseModel::Robust::shared_ptr robust_model =
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(voxel_resolution_ /
                                                           2.0),
              gausian_model);
      gtsam::LtLFactor3 factor(lidar1, lidar2, lidar3, pt_a, pt_b, robust_model,
                               1.0);

      graph_.add(factor);
    }
  }
}

void calib::ExtrinsicCalibrator::AddInitValue(const gtsam::Symbol &id,
                                       const Eigen::Isometry3d &T) {
  gtsam::Pose3 pose(T.matrix());

  initial_estimate_.insert(id, pose);
}

void calib::ExtrinsicCalibrator::AddPriorFactor(const gtsam::Symbol &id,
                                         const Eigen::Isometry3d &T) {
  gtsam::SharedDiagonal noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
  graph_.add(
      gtsam::PriorFactor<gtsam::Pose3>(id, gtsam::Pose3(T.matrix()), noise));
}

gtsam::Values calib::ExtrinsicCalibrator::SolveGraphOptimization() {
  gtsam::LevenbergMarquardtParams optimizer_params;
  optimizer_params.setLinearSolverType("MULTIFRONTAL_CHOLESKY");
  //  gtsam::LevenbergMarquardtParams::SetCeresDefaults(&optimizer_params);

  gtsam::LevenbergMarquardtOptimizer optimizer(graph_, initial_estimate_,
                                               optimizer_params);
  gtsam::Values result = optimizer.optimize();

  return result;
}

void calib::ExtrinsicCalibrator::AddBetweenFactor(const gtsam::Symbol &id1,
                                           const gtsam::Symbol &id2,
                                           const Eigen::Isometry3d &T) {
  gtsam::SharedDiagonal noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      id1, id2, gtsam::Pose3(T.matrix()), noise));
}

bool calib::ExtrinsicCalibrator::ComputeCovariancesAndPlanarity(
    const CloudPtr &cloud, nanoflann::KdTreeFLANN<PointType> &kdtree,
    std::shared_ptr<Mat4dVec> &covariances, std::vector<float> &planarity_vec,
    int knn) {
  if (kdtree.getInputCloud() != cloud) {
    kdtree.setInputCloud(cloud);
  }
  covariances->resize(cloud->size());
  planarity_vec.resize(cloud->size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    std::vector<int> k_indices;
    std::vector<float> k_sq_distances;
    kdtree.nearestKSearch(cloud->at(i), knn, k_indices, k_sq_distances);

    Eigen::Matrix<double, 4, -1> neighbors(4, knn);
    for (int j = 0; j < k_indices.size(); j++) {
      neighbors.col(j) =
          cloud->at(k_indices[j]).getVector4fMap().template cast<double>();
    }

    neighbors.colwise() -= neighbors.rowwise().mean().eval();
    Eigen::Matrix4d cov = neighbors * neighbors.transpose() / knn;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d values = Eigen::Vector3d(1, 1, 1e-3);

    (*covariances)[i].setZero();
    (*covariances)[i].template block<3, 3>(0, 0) =
        svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();

    auto eigenvalue = svd.singularValues();

    float planarity = 2 * (eigenvalue(1) - eigenvalue(2)) /
                      (eigenvalue(0) + eigenvalue(1) + eigenvalue(2));
    planarity_vec[i] = planarity;
  }

  return true;
}

void calib::ExtrinsicCalibrator::UpdateCorrespondences(
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data,
    std::vector<std::pair<int, GaussianVoxel::Ptr>> &voxel_correspondences,
    Mat4dVec &opt_cov) {
  voxel_correspondences.clear();
  auto offsets = neighbor_offsets(search_method_);

  std::vector<std::vector<std::pair<int, GaussianVoxel::Ptr>>> corrs(
      num_threads_);
  for (auto &c : corrs) {
    c.reserve((corr_pre_data.source_cloud_->size() * offsets.size()) /
              num_threads_);
  }

  CloudPtr target_cloud(new PointCloudType);
  pcl::transformPointCloud(*corr_pre_data.target_cloud_, *target_cloud,
                           corr_pre_data.target_trans_.matrix());
  Mat4dVec target_covs;
  for (size_t i = 0; i < corr_pre_data.target_cov_->size(); i++)
    target_covs.push_back(corr_pre_data.target_trans_.matrix() *
                          corr_pre_data.target_cov_->at(i) *
                          corr_pre_data.target_trans_.matrix().transpose());

  corr_pre_data.voxel_map_->create_voxelmap(*target_cloud, target_covs);

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < corr_pre_data.source_cloud_->size(); i++) {
    if (sampling_) {
      if (i % int(1 / sampling_ratio_) != 0) {
        continue;
      }
    }

    const Eigen::Vector4d mean_A = corr_pre_data.source_cloud_->at(i)
                                       .getVector4fMap()
                                       .template cast<double>();
    Eigen::Vector4d transed_mean_A = corr_pre_data.source_trans_ * mean_A;
    Eigen::Vector3i coord =
        corr_pre_data.voxel_map_->voxel_coord(transed_mean_A);

    for (const auto &offset : offsets) {
      auto voxel = corr_pre_data.voxel_map_->lookup_voxel(coord + offset);
      if (voxel != nullptr) {
        corrs[omp_get_thread_num()].push_back(std::make_pair(i, voxel));
      }
    }
  }

  voxel_correspondences.reserve(corr_pre_data.source_cloud_->size() *
                                offsets.size());
  for (const auto &c : corrs) {
    voxel_correspondences.insert(voxel_correspondences.end(), c.begin(),
                                 c.end());
  }

  opt_cov.resize(voxel_correspondences.size());

#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < voxel_correspondences.size(); i++) {
    const auto &corr = voxel_correspondences[i];
    const auto &cov_A = corr_pre_data.source_cov_->at(corr.first);
    const auto &cov_B = corr.second->cov;

    opt_cov[i] = cov_B + corr_pre_data.source_trans_.matrix() * cov_A *
                             corr_pre_data.source_trans_.matrix().transpose();
  }
}

bool calib::ExtrinsicCalibrator::IsConverged(
    const std::vector<Eigen::Isometry3d> &poses1,
    const std::vector<Eigen::Isometry3d> &poses2) {
  if (poses1.size() != poses2.size()) {
    std::cout << "poses1.size() != poses2.size()" << std::endl;
    return false;
  }

  double rotation_epsilon = 2e-3;
  double transformation_epsilon = 5e-4;

  for (size_t i = 0; i < poses1.size(); ++i) {
    Eigen::Isometry3d delta = poses1[i].inverse() * poses2[i];

    Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = delta.translation();

    Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon * R.array().abs();
    Eigen::Vector3d t_delta = 1.0 / transformation_epsilon * t.array().abs();

    double k = std::max(r_delta.maxCoeff(), t_delta.maxCoeff());
    if (k > 1.0) {
      //      std::cout << "Do not converge! k = " << k << "." << std::endl;
      return false;
    }
  }

  //  std::cout << "Converged!" << std::endl;
  return true;
}

bool calib::ExtrinsicCalibrator::IsConverged(const Eigen::Isometry3d &pose1,
                                      const Eigen::Isometry3d &pose2) {
  double rotation_epsilon = 2e-3;
  double transformation_epsilon = 5e-4;

  Eigen::Isometry3d delta = pose1.inverse() * pose2;

  Eigen::Matrix3d R = delta.linear() - Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = delta.translation();

  Eigen::Matrix3d r_delta = 1.0 / rotation_epsilon * R.array().abs();
  Eigen::Vector3d t_delta = 1.0 / transformation_epsilon * t.array().abs();

  double k = std::max(r_delta.maxCoeff(), t_delta.maxCoeff());
  if (k > 1.0) {
    //    std::cout << "Do not converge! k = " << k << "." << std::endl;
    return false;
  }
  //
  //  std::cout << "Converged!" << std::endl;
  return true;
}

void calib::ExtrinsicCalibrator::AddLgGicpFactors(
    const gtsam::Symbol &key1, const gtsam::Symbol &key2,
    const gtsam::Symbol &key3,
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data) {
  std::vector<int> correspondences;
  Mat4dVec opt_cov;
  UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

  graph_.reserve(correspondences.size());

  for (size_t i = 0; i < correspondences.size(); ++i) {
    if (correspondences[i] < 0)
      continue;

    PointType point_a = corr_pre_data.source_cloud_->points[i];
    PointType point_b = corr_pre_data.target_cloud_->points[correspondences[i]];
    Eigen::Vector3d point_a_vec(point_a.x, point_a.y, point_a.z);
    Eigen::Vector3d point_b_vec(point_b.x, point_b.y, point_b.z);

    gtsam::noiseModel::Gaussian::shared_ptr gaussian_model =
        gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
    gtsam::noiseModel::Robust::shared_ptr noise_model =
        gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(0.1), gaussian_model);
    gtsam::LgFactor3 factor(key1, key2, key3, point_a_vec, point_b_vec,
                            noise_model);

    graph_.push_back(factor);
  }
}

void calib::ExtrinsicCalibrator::AddLoopClosureFactors(const gtsam::Symbol &key1,
                                                const gtsam::Symbol &key2,
                                                const gtsam::Symbol &key3,
                                                const Eigen::Isometry3d &T) {
  gtsam::SharedDiagonal noise_model =
      gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);
  gtsam::Pose3 pose(T.matrix());
  gtsam::LgFactor4 factor(key1, key2, key3, pose, noise_model);
  graph_.push_back(factor);
}

void calib::ExtrinsicCalibrator::AddLtlFactors11(
    const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
    const gtsam::Symbol &lidar3,
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data) {
  if (huber_enable_) {
    std::vector<int> correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

    for (size_t i = 0; i < correspondences.size(); ++i) {
      if (correspondences[i] < 0) {
        continue;
      }

      PointType point_a = corr_pre_data.source_cloud_->points[i];
      PointType point_b =
          corr_pre_data.target_cloud_->points[correspondences[i]];
      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b.x, point_b.y, point_b.z);

      if (current_opt_iter_ < robust_opt_iter_) {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::noiseModel::Robust::shared_ptr robust_model =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1),
                gausian_model);
        gtsam::LtLFactor11 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                  robust_model);

        graph_.add(factor);
      } else {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::LtLFactor11 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                  gausian_model);

        graph_.add(factor);
      }
    }
  } else {
    std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, voxel_correspondences, opt_cov);

    for (size_t i = 0; i < voxel_correspondences.size(); ++i) {
      const auto &corr = voxel_correspondences[i];

      PointType point_a = corr_pre_data.source_cloud_->points[corr.first];
      Eigen::Vector4d point_b =
          corr_pre_data.target_trans_.inverse() * corr.second->mean;

      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b[0], point_b[1], point_b[2]);

      gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
          gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
      gtsam::noiseModel::Robust::shared_ptr robust_model =
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(voxel_resolution_ /
                                                           2.0),
              gausian_model);
      gtsam::LtLFactor11 factor(lidar1, lidar2, lidar3, pt_a, pt_b,
                                robust_model, 1.0);

      graph_.add(factor);
    }
  }
}

void calib::ExtrinsicCalibrator::AddLtlFactors22(
    const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data) {
  if (huber_enable_) {
    std::vector<int> correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

    for (size_t i = 0; i < correspondences.size(); ++i) {
      if (correspondences[i] < 0) {
        continue;
      }

      PointType point_a = corr_pre_data.source_cloud_->points[i];
      PointType point_b =
          corr_pre_data.target_cloud_->points[correspondences[i]];
      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b.x, point_b.y, point_b.z);

      if (current_opt_iter_ < robust_opt_iter_) {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::noiseModel::Robust::shared_ptr robust_model =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1),
                gausian_model);
        gtsam::LtLFactor22 factor(lidar1, lidar2, pt_a, pt_b, robust_model);

        graph_.add(factor);
      } else {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::LtLFactor22 factor(lidar1, lidar2, pt_a, pt_b, gausian_model);

        graph_.add(factor);
      }
    }
  } else {
    std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, voxel_correspondences, opt_cov);

    for (size_t i = 0; i < voxel_correspondences.size(); ++i) {
      const auto &corr = voxel_correspondences[i];

      PointType point_a = corr_pre_data.source_cloud_->points[corr.first];
      Eigen::Vector4d point_b =
          corr_pre_data.target_trans_.inverse() * corr.second->mean;

      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b[0], point_b[1], point_b[2]);

      gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
          gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
      gtsam::noiseModel::Robust::shared_ptr robust_model =
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(voxel_resolution_ /
                                                           2.0),
              gausian_model);
      gtsam::LtLFactor22 factor(lidar1, lidar2, pt_a, pt_b, robust_model, 1.0);

      graph_.add(factor);
    }
  }
}

void calib::ExtrinsicCalibrator::AddLtlFactors33(
    const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
    const calib::ExtrinsicCalibrator::CorrPreData &corr_pre_data) {
  if (huber_enable_) {
    std::vector<int> correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, correspondences, opt_cov);

    for (size_t i = 0; i < correspondences.size(); ++i) {
      if (correspondences[i] < 0) {
        continue;
      }

      PointType point_a = corr_pre_data.source_cloud_->points[i];
      PointType point_b =
          corr_pre_data.target_cloud_->points[correspondences[i]];
      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b.x, point_b.y, point_b.z);

      if (current_opt_iter_ < robust_opt_iter_) {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));
        gtsam::noiseModel::Robust::shared_ptr robust_model =
            gtsam::noiseModel::Robust::Create(
                gtsam::noiseModel::mEstimator::Huber::Create(0.1),
                gausian_model);
        gtsam::LtLFactor33 factor(lidar1, lidar2, pt_a, pt_b, robust_model);

        graph_.add(factor);
      } else {
        gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
            gtsam::noiseModel::Gaussian::Covariance(
                opt_cov[i].block<3, 3>(0, 0));

        gtsam::LtLFactor33 factor(lidar1, lidar2, pt_a, pt_b, gausian_model);

        graph_.add(factor);
      }
    }
  } else {
    std::vector<std::pair<int, GaussianVoxel::Ptr>> voxel_correspondences;
    Mat4dVec opt_cov;
    UpdateCorrespondences(corr_pre_data, voxel_correspondences, opt_cov);

    for (size_t i = 0; i < voxel_correspondences.size(); ++i) {
      const auto &corr = voxel_correspondences[i];

      PointType point_a = corr_pre_data.source_cloud_->points[corr.first];
      Eigen::Vector4d point_b =
          corr_pre_data.target_trans_.inverse() * corr.second->mean;

      Eigen::Vector3d pt_a(point_a.x, point_a.y, point_a.z);
      Eigen::Vector3d pt_b(point_b[0], point_b[1], point_b[2]);

      gtsam::noiseModel::Gaussian::shared_ptr gausian_model =
          gtsam::noiseModel::Gaussian::Covariance(opt_cov[i].block<3, 3>(0, 0));
      gtsam::noiseModel::Robust::shared_ptr robust_model =
          gtsam::noiseModel::Robust::Create(
              gtsam::noiseModel::mEstimator::Huber::Create(voxel_resolution_ /
                                                           2.0),
              gausian_model);
      gtsam::LtLFactor33 factor(lidar1, lidar2, pt_a, pt_b, robust_model, 1.0);

      graph_.add(factor);
    }
  }
}

void calib::ExtrinsicCalibrator::AddPlaneLgPriorFactor(
    const uint64_t &key, const Eigen::Isometry3d &T_ext) {
  double std_low = 0.001;
  double std_high = 1e8;

  gtsam::SharedDiagonal noise = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << std_high, std_high, std_high, std_high, std_high,
       std_low)
          .finished());
  graph_.add(gtsam::LgPriorFactor(gtsam::Key(key), gtsam::Pose3(T_ext.matrix()),
                                  noise));
}
