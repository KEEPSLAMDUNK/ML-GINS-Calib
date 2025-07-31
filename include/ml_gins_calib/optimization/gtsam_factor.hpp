// GTSAM factor definitions for multi-LiDAR GNSS/INS extrinsic calibration

#ifndef ML_GINS_CALIB_GTSAM_FACTOR_HPP
#define ML_GINS_CALIB_GTSAM_FACTOR_HPP

#include <iostream>

#include <gtsam/base/Matrix.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/Symbol.h>

namespace gtsam {

class MLFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;

public:
  MLFactor(gtsam::Key pose_key1, gtsam::Key pose_key2, gtsam::Vector3 point_a,
           gtsam::Vector3 point_b, gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(
            noise_model, pose_key1, pose_key2),
        point_a_(point_a), point_b_(point_b) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose1, Dpose2;
    gtsam::Point3 point_a_trans =
        pose1.transformFrom(point_a, H1 ? &Dpose1 : 0);
    gtsam::Point3 point_b_trans =
        pose2.transformFrom(point_b, H2 ? &Dpose2 : 0);

    gtsam::Vector3 residual;
    residual = point_a_trans - point_b_trans;

    if (H1) {
      H1->resize(3, 6);
      *H1 << Dpose1;
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 << -Dpose2;
    }

    return residual;
  }
};

class LgFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  gtsam::Pose3 pose_a_, pose_b_;

public:
  LgFactor(gtsam::Key pose_key, gtsam::Vector3 point_a, gtsam::Vector3 point_b,
           gtsam::Pose3 pose_a, gtsam::Pose3 pose_b,
           gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
        point_a_(point_a), point_b_(point_b), pose_a_(pose_a), pose_b_(pose_b) {
  }

  gtsam::Vector
  evaluateError(const Pose3 &pose,
                boost::optional<Matrix &> H1 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));
    gtsam::Pose3 pose_a(pose_a_);
    gtsam::Pose3 pose_b(pose_b_);

    gtsam::Matrix36 Dpose_a;
    gtsam::Matrix33 Dpoint_a;
    gtsam::Point3 point_a_trans1 =
        pose.transformFrom(point_a, H1 ? &Dpose_a : 0);
    gtsam::Point3 point_a_trans2 =
        pose_a.transformFrom(point_a_trans1, boost::none, H1 ? &Dpoint_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans1 =
        pose.transformFrom(point_b, H1 ? &Dpose_b : 0);
    gtsam::Point3 point_b_trans2 =
        pose_b.transformFrom(point_b_trans1, boost::none, H1 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = point_a_trans2 - point_b_trans2;

    if (H1) {
      H1->resize(3, 6);
      *H1 << Dpoint_a * Dpose_a - Dpoint_b * Dpose_b;
    }

    return residual;
  }
};

class LgPriorFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
private:
  gtsam::Pose3 pose_v_;

public:
  LgPriorFactor(gtsam::Key pose_key, gtsam::Pose3 pose_v,
                gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor1<gtsam::Pose3>(noise_model, pose_key),
        pose_v_(pose_v) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose,
                boost::optional<Matrix &> H1 = boost::none) const override {
    gtsam::Pose3 pose_v(pose_v_);

    gtsam::Matrix66 D_inv;
    gtsam::traits<gtsam::Pose3>::Inverse(pose, H1 ? &D_inv : 0);

    gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hlocal;
    gtsam::Vector residual = gtsam::traits<gtsam::Pose3>::Local(
        pose_v, pose.inverse(), boost::none, &Hlocal);

    if (H1) {
      H1->resize(6, 6);
      *H1 << Hlocal * D_inv;
    }

    return residual;
  }
};

class LgFactor2 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                                  gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;

public:
  LgFactor2(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3,
            gtsam::Vector3 point_a, gtsam::Vector3 point_b,
            gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
            noise_model, key1, key2, key3),
        point_a_(point_a), point_b_(point_b) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose2.transformFrom(point_a, H2 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_base1;
    gtsam::Matrix33 Dpoint_a;
    gtsam::Point3 point_a_trans2 = pose1.transformFrom(
        point_a_trans1, H1 ? &Dpose_base1 : 0, H2 ? &Dpoint_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose2.transformFrom(point_b, H2 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base2;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose3.transformFrom(
        point_b_trans1, H3 ? &Dpose_base2 : 0, H2 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = point_a_trans2 - point_b_trans2;

    if (H1) {
      H1->resize(3, 6);
      *H1 = Dpose_base1;
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 = Dpoint_a * Dpose_a - Dpoint_b * Dpose_b;
    }

    if (H3) {
      H3->resize(3, 6);
      *H3 = -Dpose_base2;
    }

    return residual;
  }
};

class LgFactor3 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                                  gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;

public:
  LgFactor3(gtsam::Symbol key1, gtsam::Symbol key2, gtsam::Symbol key3,
            gtsam::Vector3 point_a, gtsam::Vector3 point_b,
            gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
            noise_model, key1, key2, key3),
        point_a_(point_a), point_b_(point_b) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose2.transformFrom(point_a, H2 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_base1;
    gtsam::Matrix33 Dpoint_a;
    gtsam::Point3 point_a_trans2 = pose1.transformFrom(
        point_a_trans1, H1 ? &Dpose_base1 : 0, H2 ? &Dpoint_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose2.transformFrom(point_b, H2 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base2;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose3.transformFrom(
        point_b_trans1, H3 ? &Dpose_base2 : 0, H2 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = point_a_trans2 - point_b_trans2;

    if (H1) {
      H1->resize(3, 6);
      *H1 = Dpose_base1;
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 = Dpoint_a * Dpose_a - Dpoint_b * Dpose_b;
    }

    if (H3) {
      H3->resize(3, 6);
      *H3 = -Dpose_base2;
    }

    return residual;
  }
};

class LgFactor4 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                                  gtsam::Pose3> {
private:
  gtsam::Pose3 pose_delta_;

public:
  LgFactor4(gtsam::Symbol key1, gtsam::Symbol key2, gtsam::Symbol key3,
            gtsam::Pose3 pose_delta, gtsam::SharedNoiseModel noise_model)
      : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
            noise_model, key1, key2, key3),
        pose_delta_(pose_delta) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none) const override {
    gtsam::Matrix66 Dpose_1, Dpose_31;
    gtsam::Pose3 pose_a_trans =
        pose1.compose(pose3, H1 ? &Dpose_1 : 0, H3 ? &Dpose_31 : 0);

    gtsam::Matrix66 Dpose_2, Dpose_32;
    gtsam::Pose3 pose_b_trans =
        pose2.compose(pose3, H2 ? &Dpose_2 : 0, H3 ? &Dpose_32 : 0);

    gtsam::Matrix66 Dpose_a_trans, Dpose_b_trans;
    gtsam::Pose3 pose_delta_ab = gtsam::traits<gtsam::Pose3>::Between(
        pose_a_trans, pose_b_trans, &Dpose_a_trans, &Dpose_b_trans);

    gtsam::traits<gtsam::Pose3>::ChartJacobian::Jacobian Hlocal;
    gtsam::Vector residual = gtsam::traits<gtsam::Pose3>::Local(
        pose_delta_, pose_delta_ab, boost::none, &Hlocal);

    if (H1) {
      H1->resize(6, 6);
      *H1 = Hlocal * Dpose_a_trans * Dpose_1;
    }

    if (H2) {
      H2->resize(6, 6);
      *H2 = Hlocal * Dpose_b_trans * Dpose_2;
    }

    if (H3) {
      H3->resize(6, 6);
      *H3 = Hlocal * (Dpose_a_trans * Dpose_31 + Dpose_b_trans * Dpose_32);
    }

    return residual;
  }
};

class LtLFactor1 : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3,
                                                   gtsam::Pose3, gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  double w_;

  // lidar1: source pose
  // lidar2: extrinsic
  // lidar3: target pose
  // lidar4: extrinsic
public:
  LtLFactor1(gtsam::Symbol lidar1, gtsam::Symbol lidar2, gtsam::Symbol lidar3,
             gtsam::Symbol lidar4, gtsam::Vector3 point_a,
             gtsam::Vector3 point_b, gtsam::SharedNoiseModel noise_model,
             double w = 1.0)
      : gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3,
                                 gtsam::Pose3>(noise_model, lidar1, lidar2,
                                               lidar3, lidar4),
        point_a_(point_a), point_b_(point_b), w_(w) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                const Pose3 &pose4, boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none,
                boost::optional<Matrix &> H4 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose2.transformFrom(point_a, H2 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_base1;
    gtsam::Matrix33 Dpoint_a;
    gtsam::Point3 point_a_trans2 = pose1.transformFrom(
        point_a_trans1, H1 ? &Dpose_base1 : 0, H2 ? &Dpoint_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose4.transformFrom(point_b, H4 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base2;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose3.transformFrom(
        point_b_trans1, H3 ? &Dpose_base2 : 0, H4 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = point_a_trans2 - point_b_trans2;
    residual = w_ * residual;

    if (H1) {
      H1->resize(3, 6);
      *H1 << w_ * Dpose_base1;
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 << w_ * (Dpoint_a * Dpose_a);
    }

    if (H3) {
      H3->resize(3, 6);
      *H3 << w_ * (-Dpose_base2);
    }

    if (H4) {
      H4->resize(3, 6);
      *H4 << w_ * (-Dpoint_b * Dpose_b);
    }

    return residual;
  }
};

class LtLFactor2 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                                   gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  double w_;

  // lidar1: source pose
  // lidar2: extrinsic
  // lidar3: extrinsic
public:
  LtLFactor2(gtsam::Symbol lidar1, gtsam::Symbol lidar2, gtsam::Symbol lidar3,
             gtsam::Vector3 point_a, gtsam::Vector3 point_b,
             gtsam::SharedNoiseModel noise_model, double w = 1.0)
      : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
            noise_model, lidar1, lidar2, lidar3),
        point_a_(point_a), point_b_(point_b), w_(w) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose2.transformFrom(point_a, H2 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_base1;
    gtsam::Matrix33 Dpoint_a;
    gtsam::Point3 point_a_trans2 = pose1.transformFrom(
        point_a_trans1, H1 ? &Dpose_base1 : 0, H2 ? &Dpoint_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose3.transformFrom(point_b, H3 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base2;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose1.transformFrom(
        point_b_trans1, H1 ? &Dpose_base2 : 0, H3 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = w_ * (point_a_trans2 - point_b_trans2);

    if (H1) {
      H1->resize(3, 6);
      *H1 = w_ * (Dpose_base1 - Dpose_base2);
      //      *H1 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      //          std::bind(&LtLFactor2::evaluateError, this,
      //          std::placeholders::_1,
      //                    pose2, pose3, boost::none, boost::none,
      //                    boost::none),
      //          pose1);
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 = w_ * (Dpoint_a * Dpose_a);
      //      *H2 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      //          std::bind(&LtLFactor2::evaluateError, this, pose1,
      //                    std::placeholders::_1, pose3, boost::none,
      //                    boost::none, boost::none),
      //          pose2);
    }

    if (H3) {
      H3->resize(3, 6);
      *H3 = w_ * (-Dpoint_b * Dpose_b);
      //      *H3 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      //          std::bind(&LtLFactor2::evaluateError, this, pose1, pose2,
      //                    std::placeholders::_1, boost::none, boost::none,
      //                    boost::none),
      //          pose3);
    }

    return residual;
  }
};

class LtLFactor3 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                                   gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  double w_;

  // lidar1: source pose
  // lidar2: extrinsic
  // lidar3: target pose
public:
  LtLFactor3(gtsam::Symbol lidar1, gtsam::Symbol lidar2, gtsam::Symbol lidar3,
             gtsam::Vector3 point_a, gtsam::Vector3 point_b,
             gtsam::SharedNoiseModel noise_model, double w = 1.0)
      : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
            noise_model, lidar1, lidar2, lidar3),
        point_a_(point_a), point_b_(point_b), w_(w) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose2.transformFrom(point_a, H2 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_base1;
    gtsam::Matrix33 Dpoint_a;
    gtsam::Point3 point_a_trans2 = pose1.transformFrom(
        point_a_trans1, H1 ? &Dpose_base1 : 0, H2 ? &Dpoint_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose2.transformFrom(point_b, H2 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base2;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose3.transformFrom(
        point_b_trans1, H3 ? &Dpose_base2 : 0, H2 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = w_ * (point_a_trans2 - point_b_trans2);

    //    std::cout << residual.norm() << std::endl;

    if (H1) {
      H1->resize(3, 6);
      *H1 = w_ * Dpose_base1;
      //      *H1 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      //          std::bind(&LtLFactor3::evaluateError, this,
      //          std::placeholders::_1,
      //                    pose2, pose3, boost::none, boost::none,
      //                    boost::none),
      //          pose1);
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 = w_ * (Dpoint_a * Dpose_a - Dpoint_b * Dpose_b);
      //      *H2 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      //          std::bind(&LtLFactor3::evaluateError, this, pose1,
      //                    std::placeholders::_1, pose3, boost::none,
      //                    boost::none, boost::none),
      //          pose2);
    }

    if (H3) {
      H3->resize(3, 6);
      *H3 = w_ * -Dpose_base2;
      //      *H3 = gtsam::numericalDerivative11<gtsam::Vector3, gtsam::Pose3>(
      //          std::bind(&LtLFactor3::evaluateError, this, pose1, pose2,
      //                    std::placeholders::_1, boost::none, boost::none,
      //                    boost::none),
      //          pose3);
    }

    return residual;
  }
};

class LtLFactor11 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3,
                                                    gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  double w_;

  // lidar1: source pose
  // lidar2: target pose
  // lidar3: extrinsic
public:
  LtLFactor11(gtsam::Symbol lidar1, gtsam::Symbol lidar2, gtsam::Symbol lidar3,
              gtsam::Vector3 point_a, gtsam::Vector3 point_b,
              gtsam::SharedNoiseModel noise_model, double w = 1.0)
      : gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Pose3>(
            noise_model, lidar1, lidar2, lidar3),
        point_a_(point_a), point_b_(point_b), w_(w) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2, const Pose3 &pose3,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none,
                boost::optional<Matrix &> H3 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose1.transformFrom(point_a, H1 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose3.transformFrom(point_b, H3 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose2.transformFrom(
        point_b_trans1, H2 ? &Dpose_base : 0, H3 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = point_a_trans1 - point_b_trans2;
    residual = w_ * residual;

    if (H1) {
      H1->resize(3, 6);
      *H1 << w_ * Dpose_a;
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 << w_ * (-Dpose_base);
    }

    if (H3) {
      H3->resize(3, 6);
      *H3 << w_ * (-Dpoint_b * Dpose_b);
    }

    return residual;
  }
};

class LtLFactor22
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  double w_;

  // lidar1: source pose
  // lidar2: extrinsic
public:
  LtLFactor22(gtsam::Symbol lidar1, gtsam::Symbol lidar2,
              gtsam::Vector3 point_a, gtsam::Vector3 point_b,
              gtsam::SharedNoiseModel noise_model, double w = 1.0)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(noise_model,
                                                             lidar1, lidar2),
        point_a_(point_a), point_b_(point_b), w_(w) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose1.transformFrom(point_a, H1 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose2.transformFrom(point_b, H2 ? &Dpose_b : 0);

    gtsam::Matrix36 Dpose_base;
    gtsam::Matrix33 Dpoint_b;
    gtsam::Point3 point_b_trans2 = pose1.transformFrom(
        point_b_trans1, H1 ? &Dpose_base : 0, H2 ? &Dpoint_b : 0);

    gtsam::Vector3 residual;
    residual = w_ * (point_a_trans1 - point_b_trans2);

    if (H1) {
      H1->resize(3, 6);
      *H1 = w_ * (Dpose_a - Dpose_base);
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 = w_ * (-Dpoint_b * Dpose_b);
    }

    return residual;
  }
};

class LtLFactor33
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
private:
  gtsam::Vector3 point_a_, point_b_;
  double w_;

  // lidar1: source pose
  // lidar3: target pose
public:
  LtLFactor33(gtsam::Symbol lidar1, gtsam::Symbol lidar2,
              gtsam::Vector3 point_a, gtsam::Vector3 point_b,
              gtsam::SharedNoiseModel noise_model, double w = 1.0)
      : gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3>(noise_model,
                                                             lidar1, lidar2),
        point_a_(point_a), point_b_(point_b), w_(w) {}

  gtsam::Vector
  evaluateError(const Pose3 &pose1, const Pose3 &pose2,
                boost::optional<Matrix &> H1 = boost::none,
                boost::optional<Matrix &> H2 = boost::none) const override {
    gtsam::Point3 point_a(point_a_(0), point_a_(1), point_a_(2));
    gtsam::Point3 point_b(point_b_(0), point_b_(1), point_b_(2));

    gtsam::Matrix36 Dpose_a;
    gtsam::Point3 point_a_trans1 =
        pose1.transformFrom(point_a, H1 ? &Dpose_a : 0);

    gtsam::Matrix36 Dpose_b;
    gtsam::Point3 point_b_trans1 =
        pose2.transformFrom(point_b, H2 ? &Dpose_b : 0);

    gtsam::Vector3 residual;
    residual = w_ * (point_a_trans1 - point_b_trans1);

    if (H1) {
      H1->resize(3, 6);
      *H1 = w_ * Dpose_a;
    }

    if (H2) {
      H2->resize(3, 6);
      *H2 = w_ * (-Dpose_b);
    }

    return residual;
  }
};

} // namespace gtsam

#endif // ML_GINS_CALIB_GTSAM_FACTOR_HPP
