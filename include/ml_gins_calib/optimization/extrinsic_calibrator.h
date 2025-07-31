// Extrinsic calibration optimizer header for multi-LiDAR GNSS/INS system

#ifndef ML_GINS_CALIB_EXTRINSIC_CALIBRATOR_H
#define ML_GINS_CALIB_EXTRINSIC_CALIBRATOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>

#include "gtsam_factor.hpp"
#include "ml_gins_calib/nano_gicp/nanoflann.hpp"
#include "ml_gins_calib/voxel_map/vgicp_voxel.hpp"

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = pcl::PointCloud<PointType>::Ptr;

using Mat4dVec =
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>;

namespace calib {
// Extrinsic calibration optimizer for multi-LiDAR GNSS/INS system
class ExtrinsicCalibrator {
public:
  struct CorrPreData;

public:
  ExtrinsicCalibrator();
  virtual ~ExtrinsicCalibrator();

  // Configuration methods
  void SetNumThreads(int n) { num_threads_ = n; }
  void SetOptIter(int n) { opt_iter_ = n; }
  void SetSampling(bool sampling) { sampling_ = sampling; }
  void SetMaxCorrDis(double dis) { max_corr_dis_ = dis; }
  void SetSampleRatio(double ratio) { sampling_ratio_ = ratio; }
  void SetLtgPriorFactor(bool prior) { ltg_prior_factor_ = prior; }
  void SetGinsHeight(double height) {
    T_ground_imu_.setIdentity();
    T_ground_imu_.translation().z() = height;
  }

  void SolveGraphOptimization(std::vector<Eigen::Isometry3d> &res);
  void ClearGraph();

  Eigen::Isometry3d
  LgOptimizationCoarse(const std::vector<std::string> &paths,
                       const std::vector<Eigen::Isometry3d> &poses,
                       const Eigen::Isometry3d &T_ext_prior);
  Eigen::Isometry3d LgOptimizationRefine(const std::vector<std::string> &paths,
                                         std::vector<Eigen::Isometry3d> &poses,
                                         const Eigen::Isometry3d &T_ext_prior);

  Eigen::Isometry3d GroundAlignment(const CloudPtr &source_cloud,
                                    const CloudPtr &target_cloud);

  void
  JointOptimization(const std::vector<std::vector<std::string>> &cloud_paths,
                    std::vector<Eigen::Isometry3d> &gins_poses,
                    std::vector<Eigen::Isometry3d> &ltl_ext_prior,
                    const Eigen::Isometry3d &ltg_ext_prior);

  gtsam::Values SolveGraphOptimization();

private:
  void UpdateCorrespondences(const CorrPreData &corr_pre_data,
                             std::vector<int> &correspondences,
                             Mat4dVec &opt_cov);
  void UpdateCorrespondences(const CorrPreData &corr_pre_data,
                             const Eigen::Isometry3d &T_ext,
                             std::vector<int> &correspondences,
                             Mat4dVec &opt_cov);
  void UpdateCorrespondences(
      const CorrPreData &corr_pre_data,
      std::vector<std::pair<int, GaussianVoxel::Ptr>> &voxel_correspondences,
      Mat4dVec &opt_cov);

  void GroundExtraction(const CloudPtr &cloud, Eigen::Vector4d &ground_model);

  bool ComputeCovariances(const CloudPtr &cloud,
                          nanoflann::KdTreeFLANN<PointType> &kdtree,
                          std::shared_ptr<Mat4dVec> &covariances, int knn = 20);
  bool ComputeCovariancesAndPlanarity(const CloudPtr &cloud,
                                      nanoflann::KdTreeFLANN<PointType> &kdtree,
                                      std::shared_ptr<Mat4dVec> &covariances,
                                      std::vector<float> &planarity_vec,
                                      int knn = 20);

  void AddPlaneLgPriorFactor(const uint64_t &key,
                             const Eigen::Isometry3d &T_ext);
  void AddPriorFactor(const uint64_t &key, const Eigen::Isometry3d &mat);
  void AddLgPriorFactor(const uint64_t &key, const Eigen::Isometry3d &T_ext);
  void AddLgPriorFactor2(const uint64_t &key, const Eigen::Isometry3d &T);

  void AddGicpFactors(const uint64_t &key1, const uint64_t &key2,
                      const CorrPreData &corr_pre_data);
  void AddLgGicpFactors(const uint64_t &key, const CorrPreData &corr_pre_data,
                        const Eigen::Isometry3d &T_ext);
  void AddLgGicpFactors(const uint64_t &key1, const uint64_t &key2,
                        const uint64_t &key3, const CorrPreData &corr_pre_data,
                        const Eigen::Isometry3d &T_ext);
  void AddLgGicpFactors(const gtsam::Symbol &key1, const gtsam::Symbol &key2,
                        const gtsam::Symbol &key3,
                        const CorrPreData &corr_pre_data);
  void AddInitValue(const uint64_t &key, const Eigen::Isometry3d &mat);

  void AddLtlFactors1(const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
                      const gtsam::Symbol &lidar3, const gtsam::Symbol &lidar4,
                      const CorrPreData &corr_pre_data);
  void AddLtlFactors2(const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
                      const gtsam::Symbol &lidar3,
                      const CorrPreData &corr_pre_data);
  void AddLtlFactors3(const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
                      const gtsam::Symbol &lidar3,
                      const CorrPreData &corr_pre_data);

  void AddLtlFactors11(const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
                       const gtsam::Symbol &lidar3,
                       const CorrPreData &corr_pre_data);
  void AddLtlFactors22(const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
                       const CorrPreData &corr_pre_data);
  void AddLtlFactors33(const gtsam::Symbol &lidar1, const gtsam::Symbol &lidar2,
                       const CorrPreData &corr_pre_data);

  void AddLoopClosureFactors(const gtsam::Symbol &key1,
                             const gtsam::Symbol &key2,
                             const gtsam::Symbol &key3,
                             const Eigen::Isometry3d &T);

  void AddInitValue(const gtsam::Symbol &id, const Eigen::Isometry3d &T);
  void AddPriorFactor(const gtsam::Symbol &id, const Eigen::Isometry3d &T);
  void AddBetweenFactor(const gtsam::Symbol &id1, const gtsam::Symbol &id2,
                        const Eigen::Isometry3d &T);

  bool IsConverged(const std::vector<Eigen::Isometry3d> &poses1,
                   const std::vector<Eigen::Isometry3d> &poses2);
  bool IsConverged(const Eigen::Isometry3d &pose1,
                   const Eigen::Isometry3d &pose2);

public:
  struct CorrPreData {
    CloudPtr source_cloud_;
    CloudPtr target_cloud_;
    std::shared_ptr<Mat4dVec> source_cov_;
    std::shared_ptr<Mat4dVec> target_cov_;
    Eigen::Isometry3d source_trans_;
    Eigen::Isometry3d target_trans_;
    std::vector<float> source_planarity_;
    std::vector<float> target_planarity_;
    std::shared_ptr<GaussianVoxelMap<PointType>> voxel_map_;
    double corr_dist_threshold_;
  };

private:
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimate_;

  int num_threads_ = 4;
  int opt_iter_ = 10;
  double max_corr_dis_ = 1.0;
  double sampling_ratio_ = 0.01;
  bool sampling_ = true;

  bool ltg_prior_factor_ = true;

  bool huber_enable_ = true;

  int current_opt_iter_ = 0;
  int robust_opt_iter_ = 100;
  double max_error_ = 10.0;

  // voxel map
  double voxel_resolution_ = 1.0;
  NeighborSearchMethod search_method_ = NeighborSearchMethod::DIRECT1;
  VoxelAccumulationMode voxel_mode_ = VoxelAccumulationMode::ADDITIVE;

public:
  std::vector<Eigen::Isometry3d> T_exts_;

  Eigen::Isometry3d T_ground_lidar_;
  Eigen::Isometry3d T_ground_imu_;
};

} // namespace calib

#endif // ML_GINS_CALIB_EXTRINSIC_CALIBRATOR_H
