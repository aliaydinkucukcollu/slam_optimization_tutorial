#include <Eigen/Dense>
#include <Eigen/Geometry> // Eigen::Quaterniond
#include <fstream>
#include <iostream>

#include <ceres/ceres.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

// 3D Pose => rotation + translation
struct Pose3D {
  Eigen::Quaterniond q; // rotation
  Eigen::Vector3d t;    // translation
};

/**
    compute relative pose from i to j
    Ti --------------------------> Tj
                  T_ij

    T_ij = Ti^{-1} * Tj

    q_ij = q_i^{-1} * q_j
    t_ij = R_i^T * (t_j - t_i)
*/
Pose3D getRelativePose(const Pose3D &Ti, const Pose3D &Tj) {
  Pose3D Tij;

  Eigen::Quaterniond qi_Inverse = Ti.q.conjugate();

  Tij.q = (qi_Inverse * Tj.q).normalized();
  Tij.t = qi_Inverse * (Tj.t - Ti.t);

  return Tij;
}

void printPose(const Pose3D &T) {
  std::cout << "q => x: " << T.q.x() << " y: " << T.q.y() << " z: " << T.q.z()
            << " w: " << T.q.w() << '\n';
  std::cout << "t => x: " << T.t.x() << " y: " << T.t.y() << " z: " << T.t.z()
            << "\n\n";
}

int main() {

  Pose3D Ti;
  Pose3D Tj;

  // save poses to visualize with python
  std::ofstream out;
  out.open("/home/aliaydin/Desktop/awesome_cpp/slam_optimization_tutorial/data/"
           "Ti.csv");

  Ti.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
  Ti.t = Eigen::Vector3d(1.0, 2.0, 1.5);
  printPose(Ti);
  out << Ti.q.w() << ", " << Ti.q.x() << ", " << Ti.q.y() << ", " << Ti.q.z()
      << ", " << Ti.t.x() << ", " << Ti.t.y() << ", " << Ti.t.z() << "\n";
  out.close();

  out.open("/home/aliaydin/Desktop/awesome_cpp/slam_optimization_tutorial/data/"
           "Tj.csv");
  Tj.q = Eigen::Quaterniond(Eigen::AngleAxisd(0.8, Eigen::Vector3d::UnitZ()));
  Tj.t = Eigen::Vector3d(3.0, 1.0, 0.7);
  printPose(Tj);
  out << Tj.q.w() << ", " << Tj.q.x() << ", " << Tj.q.y() << ", " << Tj.q.z()
      << ", " << Tj.t.x() << ", " << Tj.t.y() << ", " << Tj.t.z() << "\n";
  out.close();

  out.open("/home/aliaydin/Desktop/awesome_cpp/slam_optimization_tutorial/data/"
           "Tij.csv");
  Pose3D Tij = getRelativePose(Ti, Tj);
  printPose(Tij);
  out << Tij.q.w() << ", " << Tij.q.x() << ", " << Tij.q.y() << ", "
      << Tij.q.z() << ", " << Tij.t.x() << ", " << Tij.t.y() << ", "
      << Tij.t.z() << "\n";
  out.close();

  return 0;
}