/*
 * Description:
 * Created by WJH on 2023/7/10.
 */

#include "ml_gins_calib/hand_eye_calib/EigenUtils.h"
#include "ml_gins_calib/hand_eye_calib/HandEyeCalibration.h"

#include <fstream>
#include <iostream>

typedef Eigen::Matrix4d Pose;
typedef std::vector<Pose> Poses;

void read_poses(const std::string path1, const std::string path2, Poses &A,
                Poses &B);

int main(int argc, char **argv) {
  Poses A;
  Poses B;

  std::string path1 = argv[1];
  std::string path2 = argv[2];
  read_poses(path1, path2, A, B);

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      rvecs1, tvecs1, rvecs2, tvecs2;

  for (size_t i = 0; i < A.size(); ++i) {
    Eigen::Vector3d rvec1, tvec1, rvec2, tvec2;
    Eigen::AngleAxisd angle_axis1(A[i].block<3, 3>(0, 0));
    rvec1 = angle_axis1.angle() * angle_axis1.axis();
    tvec1 = A[i].block<3, 1>(0, 3);
    Eigen::AngleAxisd angle_axis2(B[i].block<3, 3>(0, 0));
    rvec2 = angle_axis2.angle() * angle_axis2.axis();
    tvec2 = B[i].block<3, 1>(0, 3);

    rvecs1.push_back(rvec1);
    tvecs1.push_back(tvec1);
    rvecs2.push_back(rvec2);
    tvecs2.push_back(tvec2);
  }

  Eigen::Matrix4d T;
  wts_sensor_calib::HandEyeCalibration::estimateHandEyeScrew(
      rvecs1, tvecs1, rvecs2, tvecs2, T, true);

  std::cout << std::endl;
  std::cout << "*************************" << std::endl;

  Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Eigen::Quaterniond q(T.block<3, 3>(0, 0));
  std::cout << t.x() << " " << t.y() << " " << t.z() << " " << q.x() << " "
            << q.y() << " " << q.z() << " " << q.w() << std::endl;

  return 0;
}

void read_poses(const std::string path1, const std::string path2, Poses &A,
                Poses &B) {
  std::ifstream in_file1;
  std::ifstream in_file2;
  in_file1.open(path1);
  in_file2.open(path2);

  if (!in_file1 || !in_file2) {
    std::cout << "Unable to open file!";
    exit(1);
  }

  Pose T_last = Pose::Identity();
  bool b_init = false;

  int count = 0;
  std::string line;
  while (getline(in_file1, line)) {
    count++;
    if (count < 1)
      continue;
    std::stringstream ss(line);
    Pose pose_matrix = Pose::Identity();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        ss >> pose_matrix(i, j);

    if (!b_init) {
      T_last = pose_matrix;
      b_init = true;
    }
    auto delta_T = T_last.inverse() * pose_matrix;

    //      std::cout << pose_matrix(2, 3) << std::endl;

    A.push_back(delta_T);

    count = 0;
  }
  b_init = false;

  count = 0;
  while (getline(in_file2, line)) {
    count++;
    if (count < 1)
      continue;

    std::stringstream ss(line);
    Pose pose_matrix = Pose::Identity();
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
        ss >> pose_matrix(i, j);

    // test
    //    Pose matrix = Pose::Identity();
    //    matrix(0, 0) = -1;
    //    matrix(1, 1) = -1;
    //    matrix(2, 2) = -1;
    //    matrix(0, 3) = 2;
    //    matrix(1, 3) = 2;
    //    pose_matrix = pose_matrix * matrix;

    if (!b_init) {
      T_last = pose_matrix;
      b_init = true;
    }
    auto delta_T = T_last.inverse() * pose_matrix;

    //      std::cout << pose_matrix(2, 3) << std::endl;

    B.push_back(delta_T);

    count = 0;
  }

  if (A.size() != B.size()) {
    std::cout << "Unequal size: " << A.size() << "  " << B.size() << std::endl;
    exit(1);
  }

  in_file1.close();
  in_file2.close();
}