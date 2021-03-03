#include "ematrix.h"

#include "defines.h"
#include "fmatrix.h"
#include "triangulation.h"
#include "sfm_utils.h"

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <iostream>

namespace {

// essential matrix must have exactly two equal non zero singular values
// (см. Hartley & Zisserman p.257)
void ensureSpectralProperty(matrix3d &Ecv)
{
    Eigen::MatrixXd E;
    copy(Ecv, E);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3d iden = svd.singularValues().asDiagonal();
    iden(0,0) = iden(1,1);
    iden(2,2) = 0.;

    Eigen::MatrixXd U = svd.matrixU();
    //        U.block<2,2>(1,1) = Eigen::Matrix2d::Identity();
    //        Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    //        identity(2,2) = 0;
    //        identity.block<2,2>(1,1) = Eigen::Matrix2d::Identity().transpose();

    E = U * iden * svd.matrixV().transpose();

    //        throw std::runtime_error("not implemented yet");
    // TODO

    copy(E, Ecv);
}

}

cv::Matx33d phg::fmatrix2ematrix(const cv::Matx33d &F, const phg::Calibration &calib0, const phg::Calibration &calib1)
{
    //    throw std::runtime_error("not implemented yet");
    matrix3d E = calib1.K().t() * F * calib0.K();

    ensureSpectralProperty(E);

    return E;
}

namespace {

matrix34d composeP(const Eigen::MatrixXd &R, const Eigen::VectorXd &t)
{
    matrix34d result;

    result(0, 0) = R(0, 0);
    result(0, 1) = R(0, 1);
    result(0, 2) = R(0, 2);
    result(1, 0) = R(1, 0);
    result(1, 1) = R(1, 1);
    result(1, 2) = R(1, 2);
    result(2, 0) = R(2, 0);
    result(2, 1) = R(2, 1);
    result(2, 2) = R(2, 2);

    result(0, 3) = t[0];
    result(1, 3) = t[1];
    result(2, 3) = t[2];

    return result;
}

double getDepth(const vector2d &m0, const vector2d &m1, const phg::Calibration &calib0, const phg::Calibration &calib1, const matrix34d &P0, const matrix34d &P1)
{
    throw std::runtime_error("not implemented yet");
    vector3d p0 = vector3d(m0(0), m0(1), 1.);
    vector3d p1 = vector3d(m0(1), m1(1), 1.);

    vector3d ps[2] = {p0, p1};
    matrix34d Ps[2] = {P0, P1};

    vector4d X = phg::triangulatePoint(Ps, ps, 2);
    if (X[3] != 0) {
        X /= X[3];
    }

    return TODO;
}
}

// Матрицы камер для фундаментальной матрицы определены с точностью до проективного преобразования
// То есть, можно исказить трехмерный мир (применив 4-мерную однородную матрицу), и одновременно поменять матрицы P0, P1 так, что проекции в пикселях не изменятся
// Если мы знаем калибровки камер (матрицы K0, K1 в структуре матриц P0, P1), то можем наложить дополнительные ограничения, в частности, известно, что
// существенная матрица (Essential matrix = K1t * F * K0) имеет ровно два совпадающих ненулевых сингулярных значения, тогда как для фундаментальной матрицы они могут различаться
// Это дополнительное ограничение позволяет разложить существенную матрицу с точностью до 4 решений, вместо произвольного проективного преобразования (см. Hartley & Zisserman p.258)
// Обычно мы можем использовать одну общую калибровку, более менее верную для большого количества реальных камер и с ее помощью выполнить
// первичное разложение существенной матрицы (а из него, взаимное расположение камер) для последующего уточнения методом нелинейной оптимизации
void phg::decomposeEMatrix(cv::Matx34d &P0, cv::Matx34d &P1, const cv::Matx33d &Ecv, const std::vector<cv::Vec2d> &m0, const std::vector<cv::Vec2d> &m1, const Calibration &calib0, const Calibration &calib1)
{
    //    throw std::runtime_error("not implemented yet");
    if (m0.size() != m1.size()) {
        throw std::runtime_error("decomposeEMatrix : m0.size() != m1.size()");
    }

    using mat = Eigen::MatrixXd;
    using vec = Eigen::VectorXd;

    mat E;
    copy(Ecv, E);

    // (см. Hartley & Zisserman p.258)

    Eigen::JacobiSVD<mat> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    mat U = svd.matrixU();
    vec s = svd.singularValues();
    mat V = svd.matrixV();

    // U, V must be rotation matrices, not just orthogonal
    if (U.determinant() < 0) U = -U;
    if (V.determinant() < 0) V = -V;

    std::cout << "U:\n" << U << std::endl;
    std::cout << "s:\n" << s << std::endl;
    std::cout << "V:\n" << V << std::endl;

    Eigen::Matrix3d w = Eigen::Matrix3d::Zero();
    w(0,1) = -1.;
    w(1,0) = 1.;
    w(2,2) = 1.;


    mat R0 = U * w.transpose() * V.transpose();
    mat R1 = U * w * V.transpose();

    std::cout << "R0:\n" << R0 << std::endl;
    std::cout << "R1:\n" << R1 << std::endl;

    vec t0 = U.col(U.cols()-1);
    vec t1 = U.col(U.cols()-1) * (-1.);

    std::cout << "t0:\n" << t0 << std::endl;

    P0 = matrix34d::eye();

    // 4 possible solutions
    matrix34d P10 = composeP(R0, t0);
    matrix34d P11 = composeP(R0, t1);
    matrix34d P12 = composeP(R1, t0);
    matrix34d P13 = composeP(R1, t1);
    matrix34d P1s[4] = {P10, P11, P12, P13};

    // need to select best of 4 solutions: 3d points should be in front of cameras (positive depths)
    int best_count = 0;
    int best_idx = -1;
    for (int i = 0; i < 4; ++i) {
        int count = 0;
        for (int j = 0; j < (int) m0.size(); ++j) {


            //https://stackoverflow.com/questions/22807039/decomposition-of-essential-matrix-validation-of-the-four-possible-solutions-for
            cv::Matx33d R; cv::Vec3d O;
            decomposeUndistortedPMatrix(R,O,P1s[i]);

            matrix3d M = skew(O);

            cv::Vec3d m03d = {m0.at(j)(0), m0.at(j)(1), 1.};
            cv::Vec3d m13d = {m1.at(j)(0), m1.at(j)(1), 1.};

            cv::Vec3d p0 = M * calib0.K().inv() * m03d;
            cv::Vec3d p1 = M * R.t() * calib1.K().inv()*m13d;

            std::cout << p0 << " " << p1 << std::endl;
            if (p0(2) * p1(2) > 0)
            {
                ++count;
            }
        }
        std::cout << "decomposeEMatrix: count: " << count << std::endl;
        if (count > best_count) {
            best_count = count;
            best_idx = i;
        }
    }

    if (best_count == 0) {
        throw std::runtime_error("decomposeEMatrix : can't decompose ematrix");
    }

    P1 = P1s[best_idx];

    std::cout << "best idx: " << best_idx << std::endl;
    std::cout << "P0: \n" << P0 << std::endl;
    std::cout << "P1: \n" << P1 << std::endl;
}

void phg::decomposeUndistortedPMatrix(cv::Matx33d &R, cv::Vec3d &O, const cv::Matx34d &P)
{
    R = P.get_minor<3, 3>(0, 0);

    cv::Matx31d O_mat = -R.t() * P.get_minor<3, 1>(0, 3);
    O(0) = O_mat(0);
    O(1) = O_mat(1);
    O(2) = O_mat(2);

    if (cv::determinant(R) < 0) {
        R *= -1;
    }
}

cv::Matx33d phg::composeEMatrixRT(const cv::Matx33d &R, const cv::Vec3d &T)
{
    return skew(T) * R;
}

cv::Matx34d phg::composeCameraMatrixRO(const cv::Matx33d &R, const cv::Vec3d &O)
{
    vector3d T = -R * O;
    return make34(R, T);
}
