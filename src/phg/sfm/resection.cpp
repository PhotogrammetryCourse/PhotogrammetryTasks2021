#include "resection.h"

#include <Eigen/SVD>
#include <iostream>
#include "sfm_utils.h"
#include "defines.h"

namespace {

// Сделать из первого минора 3х3 матрицу вращения, скомпенсировать масштаб у компоненты сдвига
matrix34d canonicalizeP(const matrix34d &P)
{
    matrix3d RR = P.get_minor<3, 3>(0, 0);
    vector3d tt;
    tt[0] = P(0, 3);
    tt[1] = P(1, 3);
    tt[2] = P(2, 3);

    if (cv::determinant(RR) < 0) {
        RR *= -1;
        tt *= -1;
    }

    double sc = 0;
    for (int i = 0; i < 9; i++) {
        sc += RR.val[i] * RR.val[i];
    }
    sc = std::sqrt(3 / sc);

    Eigen::MatrixXd RRe;
    copy(RR, RRe);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(RRe, Eigen::ComputeFullU | Eigen::ComputeFullV);
    RRe = svd.matrixU() * svd.matrixV().transpose();
    copy(RRe, RR);

    tt *= sc;

    matrix34d result;
    for (int i = 0; i < 9; ++i) {
        result(i / 3, i % 3) = RR(i / 3, i % 3);
    }
    result(0, 3) = tt(0);
    result(1, 3) = tt(1);
    result(2, 3) = tt(2);

    return result;
}

// (см. Hartley & Zisserman p.178)
cv::Matx34d estimateCameraMatrixDLT(const cv::Vec3d *Xs, const cv::Vec3d *xs, int count)
{
    //        throw std::runtime_error("not implemented yet");
    using mat = Eigen::MatrixXd;
    using vec = Eigen::VectorXd;

    mat A(11,12);

    for (int i = 0; i < count; ++i) {

        double x = xs[i][0];
        double y = xs[i][1];
        double w = xs[i][2];

        double X = Xs[i][0];
        double Y = Xs[i][1];
        double Z = Xs[i][2];
        double W = 1.0;

        A.row(i * 2) << 0.,0.,0.,0.,-w*X,-w*Y,-w*Z,-w*W,y*X,y*Y,y*Z,y*W;
        if(i * 2 + 1 != 11)
            A.row(i * 2 + 1) << w*X,w*Y,w*Z,w*W,0.,0.,0.,0.,-x*X,-x*Y,-x*Z,-x*W;
    }

    matrix34d result;

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd null_space = svda.matrixV().col(svda.matrixV().cols()-1);

    result = matrix34d(null_space[0], null_space[1], null_space[2], null_space[3],
            null_space[4], null_space[5], null_space[6], null_space[7],
            null_space[8], null_space[9], null_space[10], null_space[11]);
    return canonicalizeP(result);
}


// По трехмерным точкам и их проекциям на изображении определяем положение камеры
cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
{
    //        throw std::runtime_error("not implemented yet");
    if (X.size() != x.size()) {
        throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
    }

    const int n_points = X.size();

    // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
    // будет отличаться от случая с гомографией
    const int n_trials = 10000;//TODO

    const double threshold_px = 3;

    const int n_samples = 6;
    uint64_t seed = 1;

    int best_support = 0;
    cv::Matx34d best_P;

    std::vector<int> sample;
    for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
        phg::randomSample(sample, n_points, n_samples, &seed);

        cv::Vec3d ms0[n_samples];
        cv::Vec3d ms1[n_samples];
        for (int i = 0; i < n_samples; ++i) {
            ms0[i] = X[sample[i]];//TODO;
            ms1[i] = calib.unproject(x[sample[i]]);//TODO;
        }

        cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

        int support = 0;
        for (int i = 0; i < n_points; ++i) {
            cv::Vec3d d3 = calib.project(P * to_homogenus(X[i]));
//            std::cout << "d3 = " << d3 << " " << d3 / d3(2) <<std::endl;
            cv::Vec2d px = from_homogenus(d3 / d3(2));//TODO спроецировать 3Д точку в пиксель с использованием P и calib;
            if (cv::norm(px - x[i]) < threshold_px) {
                ++support;
            }
        }

        if (support > best_support) {
            best_support = support;
            best_P = P;

            std::cout << "estimateCameraMatrixRANSAC : support: " << best_support << "/" << n_points << std::endl;

            if (best_support == n_points) {
                break;
            }
        }
    }

    std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

    if (best_support == 0) {
        throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
    }

    return best_P;
}


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
