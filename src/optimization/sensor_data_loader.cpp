// Sensor data loader implementation for multi-LiDAR GNSS/INS calibration

#include <filesystem>
#include <fstream>
#include <iostream>

#include "ml_gins_calib/optimization/sensor_data_loader.h"

calib::DataLoader::DataLoader(const SensorList &sensor_list)
    : sensor_list_(sensor_list) {}
calib::DataLoader::~DataLoader() {}

bool calib::DataLoader::LoadSensorList(const std::string &config_path,
                                       SensorList &sensor_list) {
  try {
    YAML::Node config_node = YAML::LoadFile(config_path);
    YAML::Node lidar_list_node = config_node["lidar_list"];
    for (auto it = lidar_list_node.begin(); it != lidar_list_node.end(); ++it) {
      sensor_list.lidar_list_.emplace_back(it->as<std::string>());
    }

    sensor_list.gnss_ins_ = config_node["gnss/ins"].as<std::string>();
    sensor_list.data_dir_ = config_node["data_dir"].as<std::string>();
    sensor_list.data_mask_path_ = config_node["data_mask_path"].as<std::string>();

    return true;
  } catch (const YAML::BadFile &e) {
    std::cerr << "Error reading config file: " << e.what() << std::endl;
    return false;
  } catch (const YAML::Exception &e) {
    std::cerr << "Error parsing config file: " << e.what() << std::endl;
    return false;
  }
}

bool calib::DataLoader::LoadLtgExtInfo(const std::string &config_path) {
  try {
    YAML::Node config_node = YAML::LoadFile(config_path);
    YAML::Node ltg_ext_list_node = config_node["init_ext"];

    std::map<std::string, LtgExtInfo> lidar_ext_map;
    for (auto it = ltg_ext_list_node.begin(); it != ltg_ext_list_node.end();
         ++it) {
      LtgExtInfo ext_info;
      ext_info.lidar_name_ = it->first.as<std::string>();
      YAML::Node ext_q_node =
          ltg_ext_list_node[ext_info.lidar_name_]["orientation"];
      YAML::Node ext_t_node =
          ltg_ext_list_node[ext_info.lidar_name_]["position"];

      Eigen::Quaterniond q;
      Eigen::Vector3d t;
      q.w() = ext_q_node["w"].as<double>();
      q.x() = ext_q_node["x"].as<double>();
      q.y() = ext_q_node["y"].as<double>();
      q.z() = ext_q_node["z"].as<double>();
      t.x() = ext_t_node["x"].as<double>();
      t.y() = ext_t_node["y"].as<double>();
      t.z() = ext_t_node["z"].as<double>();

      q.normalize();

      ext_info.T_ext_.setIdentity();
      ext_info.T_ext_.linear() = q.toRotationMatrix();
      ext_info.T_ext_.translation() = t;

      lidar_ext_map[ext_info.lidar_name_] = ext_info;
    }

    for (const auto &lidar_name : sensor_list_.lidar_list_) {
      if (lidar_ext_map.count(lidar_name)) {
        ltg_ext_infos_.emplace_back(lidar_ext_map[lidar_name]);
      } else {
        ltg_ext_infos_.clear();
        std::cerr << "The number of lidar list and init ext info is not equal!"
                  << std::endl;
        return false;
      }
    }

    return true;
  } catch (const YAML::BadFile &e) {
    std::cerr << "Error reading config file: " << e.what() << std::endl;
    return false;
  } catch (const YAML::Exception &e) {
    std::cerr << "Error parsing config file: " << e.what() << std::endl;
    return false;
  }
}

bool calib::DataLoader::LoadLidarData() {
  lidar_data_list_.clear();

  namespace fs = std::filesystem;

  auto compare_filenames = [](const std::string &filename1,
                              const std::string &filename2) -> bool {
    // Extract numeric part from filename for chronological sorting
    double num1 = std::stod(filename1.substr(
        filename1.rfind("/") + 1, filename1.rfind('.') - filename1.rfind("/")));
    double num2 = std::stod(filename2.substr(
        filename2.rfind("/") + 1, filename2.rfind('.') - filename2.rfind("/")));
    return num1 < num2;
  };

  for (auto it = sensor_list_.lidar_list_.begin();
       it != sensor_list_.lidar_list_.end(); ++it) {
    std::string data_dir = sensor_list_.data_dir_ + "/" + *it;

    LidarData lidar_data;
    lidar_data.lidar_name_ = *it;
    for (const fs::directory_entry &entry : fs::directory_iterator(data_dir)) {
      if (entry.path().extension() == ".pcd") {
        lidar_data.data_list_.emplace_back(entry.path());
      }
    }
    std::sort(lidar_data.data_list_.begin(), lidar_data.data_list_.end(),
              compare_filenames);

    if (lidar_data.data_list_.size() != sync_data_size_) {
      lidar_data_list_.clear();
      std::cout << "The number of synchronous data is not equal!" << std::endl;
      return false;
    }

    lidar_data_list_.push_back(lidar_data);
  }

  if (sensor_list_.lidar_list_.size() != lidar_data_list_.size()) {
    lidar_data_list_.clear();
    std::cerr << "The number of lidar list and lidar data list is not equal!"
              << std::endl;
    return false;
  }

  return true;
}

bool calib::DataLoader::LoadGnssInsData() {
  gnss_ins_data_.clear();

  namespace fs = std::filesystem;

  std::string data_dir = sensor_list_.data_dir_ + "/" + sensor_list_.gnss_ins_;
  std::string data_path;
  for (const fs::directory_entry &entry : fs::directory_iterator(data_dir)) {
    if (entry.path().extension() == ".txt") {
      data_path = entry.path();
    }
  }

  std::ifstream infile(data_path);
  if (infile.is_open()) {
    std::string line;
    while (std::getline(infile, line)) {
      std::stringstream ss(line);
      std::string timestamp;
      ss >> timestamp;

      Eigen::Isometry3d Tg = Eigen::Isometry3d::Identity();
      ss >> Tg(0, 0) >> Tg(0, 1) >> Tg(0, 2) >> Tg(0, 3) >> Tg(1, 0) >>
          Tg(1, 1) >> Tg(1, 2) >> Tg(1, 3) >> Tg(2, 0) >> Tg(2, 1) >>
          Tg(2, 2) >> Tg(2, 3);

      gnss_ins_data_.emplace_back(Tg);
    }
    infile.close();
  } else {
    std::cerr << "Cannot open GNSS/INS data file" << std::endl;
  }

  infile.close();

  sync_data_size_ = gnss_ins_data_.size();

  return true;
}

bool calib::DataLoader::LoadCalibData() {
  if (!LoadGnssInsData())
    return false;

  if (!LoadLidarData())
    return false;

  return true;
}

void calib::DataLoader::SetTrajectoryOrigin(
    const Eigen::Isometry3d &T_ori, std::vector<Eigen::Isometry3d> &T_vec) {
  Eigen::Isometry3d T_ori_inv = T_ori.inverse();
  for (size_t i = 0; i < T_vec.size(); ++i) {
    T_vec[i] = T_ori_inv * T_vec[i];
  }
}

bool calib::DataLoader::LoadGinsInstallationHeight(
    const std::string &config_path) {
  try {
    YAML::Node config_node = YAML::LoadFile(config_path);
    gins_installation_height_ = config_node["gins_installation_height"].as<double>();
    return true;
  } catch (const YAML::BadFile &e) {
    std::cerr << "Error reading config file: " << e.what() << std::endl;
    return false;
  } catch (const YAML::Exception &e) {
    std::cerr << "Error parsing config file: " << e.what() << std::endl;
    return false;
  }
}


