// Sensor data loader header for multi-LiDAR GNSS/INS calibration

#ifndef ML_GINS_CALIB_SENSOR_DATA_LOADER_H
#define ML_GINS_CALIB_SENSOR_DATA_LOADER_H

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

namespace calib {
// Sensor data loader for multi-LiDAR GNSS/INS calibration system
class DataLoader {
public:
  struct SensorList;
  struct LtgExtInfo;
  struct LidarData;

public:
  DataLoader(const SensorList &sensor_list);
  virtual ~DataLoader();

  static bool LoadSensorList(const std::string &config_path,
                             SensorList &sensor_list);

  bool LoadLtgExtInfo(const std::string &config_path);

  bool LoadCalibData();

  bool LoadGinsInstallationHeight(const std::string &config_path);

  void SetTrajectoryOrigin(const Eigen::Isometry3d &T_ori,
                           std::vector<Eigen::Isometry3d> &T_vec);

private:
  bool LoadLidarData();
  bool LoadGnssInsData();

public:
  struct SensorList {
    std::vector<std::string> lidar_list_;  // List of LiDAR sensor names
    std::string gnss_ins_;                 // GNSS/INS sensor name
    std::string data_dir_;                 // Base data directory
    std::string data_mask_path_;           // Data mask file path
  };

  struct LtgExtInfo {
    std::string lidar_name_;     // LiDAR sensor name
    Eigen::Isometry3d T_ext_;    // Extrinsic transformation matrix
  };

  struct LidarData {
    std::string lidar_name_;               // LiDAR sensor name
    std::vector<std::string> data_list_;   // List of point cloud file paths
    std::vector<bool> data_mask_;          // Data selection mask
  };

  std::vector<LtgExtInfo> ltg_ext_infos_;
  std::vector<LidarData> lidar_data_list_;
  std::vector<Eigen::Isometry3d> gnss_ins_data_;

private:
  SensorList sensor_list_;
  size_t sync_data_size_ = -1;

public:
  double gins_installation_height_ = 0.0;
};
} // namespace calib

#endif // ML_GINS_CALIB_SENSOR_DATA_LOADER_H
