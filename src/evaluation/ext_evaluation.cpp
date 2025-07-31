/* 
 * Description:
 * Created by WJH on 23-12-13.
 */

#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include "manif/SO3.h"

struct ExtMat {
  Eigen::Quaterniond q;
  Eigen::Vector3d t;
};

std::vector<ExtMat> LoadExtMat(const std::string &path);

int main(int argc, char **argv) {
  if (true) {
    std::vector<ExtMat> eva_ext_mats = LoadExtMat(argv[1]);
    std::vector<ExtMat> gt_ext_mats = LoadExtMat(argv[2]);

    size_t size = eva_ext_mats.size();

    std::cout << size << std::endl;

    std::vector<std::pair<double, double>> errs;
    std::vector<std::vector<double>> errs_6d;
    for (size_t i = 0; i < size; ++i) {
      manif::SO3d R_eva(eva_ext_mats[i].q);
      manif::SO3d R_gt(gt_ext_mats[i].q);

      Eigen::Vector3d t_eva = eva_ext_mats[i].t;
      Eigen::Vector3d t_gt = gt_ext_mats[i].t;

      auto R_diff = R_gt.inverse() * R_eva;
      auto R_diff_so3 = R_diff.log();
      auto t_diff = t_gt - t_eva;

      std::pair<double, double> err;
      err.first = sqrt(R_diff_so3.coeffs().transpose() * R_diff_so3.coeffs()) / M_PI * 180;
      err.second = sqrt(t_diff.transpose() * t_diff);
      errs.emplace_back(err);

      std::vector<double> err_6d =
          {R_diff_so3.x() / M_PI * 180, R_diff_so3.y() / M_PI * 180, R_diff_so3.z() / M_PI * 180, t_diff[0], t_diff[1],
           t_diff[2]};
      errs_6d.emplace_back(err_6d);
    }

    std::ofstream outfile(argv[3]);
    for (const auto &item: errs)
      outfile << item.first << " " << item.second << std::endl;
    for (const auto &item: errs_6d)
      outfile << item[0] << " " << item[1] << " " << item[2] << " " << item[3] << " " << item[4] << " " << item[5]
              << std::endl;
    outfile.close();
  } else {
    std::vector<ExtMat> eva_ext_mats = LoadExtMat(argv[1]);

    std::vector<std::vector<double>> eva_ext_6d;
    for (const auto &T_ext: eva_ext_mats) {
      manif::SO3d R_eva(T_ext.q);
      Eigen::Vector3d t_eva = T_ext.t;

      auto R_eva_so3 = R_eva.log();

//      std::vector<double> eva_ext =
//          {R_eva_so3[0] / M_PI * 180, R_eva_so3[1] / M_PI * 180, R_eva_so3[2] / M_PI * 180, t_eva[0], t_eva[1],
//           t_eva[2]};
      std::vector<double> eva_ext =
          {R_eva_so3[0], R_eva_so3[1], R_eva_so3[2], t_eva[0], t_eva[1], t_eva[2]};
      eva_ext_6d.emplace_back(eva_ext);
    }

    std::ofstream outfile(argv[2]);
    for (const auto &item: eva_ext_6d)
      outfile << item[0] << " " << item[1] << " " << item[2] << " " << item[3] << " " << item[4] << " " << item[5]
              << std::endl;
    outfile.close();
  }

  return 0;
}

std::vector<ExtMat> LoadExtMat(const std::string &path) {
  std::fstream infile(path);

  std::string line;
  std::vector<ExtMat> ext_mats;
  while (getline(infile, line)) {
    std::stringstream ss(line);
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    ss >> t.x() >> t.y() >> t.z();
    ss >> q.x() >> q.y() >> q.z() >> q.w();

    ExtMat ext_mat;
    ext_mat.q = q;
    ext_mat.t = t;

    ext_mats.emplace_back(ext_mat);
  }

  infile.close();

  return ext_mats;
}