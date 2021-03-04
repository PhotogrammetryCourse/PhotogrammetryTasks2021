#pragma once

#include <opencv2/core.hpp>

typedef cv::Matx22d matrix2d;
typedef cv::Vec2d vector2d;
typedef cv::Matx33d matrix3d;
typedef cv::Matx34d matrix34d;
typedef cv::Vec3d vector3d;
typedef cv::Matx44d matrix4d;
typedef cv::Vec4d vector4d;

inline matrix3d skew(const vector3d &m)
{
    matrix3d result;

    double x = m[0];
    double y = m[1];
    double z = m[2];

    // 0, -z, y
    result(0, 0) = 0;
    result(0, 1) = -z;
    result(0, 2) = y;

    // z, 0, -x
    result(1, 0) = z;
    result(1, 1) = 0;
    result(1, 2) = -x;

    // -y, x, 0
    result(2, 0) = -y;
    result(2, 1) = x;
    result(2, 2) = 0;

    return result;
}

inline matrix34d make34(const matrix3d &R, const vector3d &O)
{
    matrix34d result;
    for (int i = 0; i < 9; ++i) {
        result(i / 3, i % 3) = R(i / 3, i % 3);
    }
    for (int i = 0; i < 3; ++i) {
        result(i, 3) = O(i);
    }
    return result;
}

template <typename EIGEN_TYPE>
inline void copy(const matrix3d &Fcv, EIGEN_TYPE &F)
{
    F = EIGEN_TYPE(3, 3);

    F(0, 0) = Fcv(0, 0); F(0, 1) = Fcv(0, 1); F(0, 2) = Fcv(0, 2);
    F(1, 0) = Fcv(1, 0); F(1, 1) = Fcv(1, 1); F(1, 2) = Fcv(1, 2);
    F(2, 0) = Fcv(2, 0); F(2, 1) = Fcv(2, 1); F(2, 2) = Fcv(2, 2);
}

template <typename EIGEN_TYPE>
inline void copy(const matrix4d &Fcv, EIGEN_TYPE &F)
{
    F = EIGEN_TYPE(4, 4);

    F(0, 0) = Fcv(0, 0); F(0, 1) = Fcv(0, 1); F(0, 2) = Fcv(0, 2);F(0, 3) = Fcv(0, 3);
    F(1, 0) = Fcv(1, 0); F(1, 1) = Fcv(1, 1); F(1, 2) = Fcv(1, 2);F(1, 3) = Fcv(1, 3);
    F(2, 0) = Fcv(2, 0); F(2, 1) = Fcv(2, 1); F(2, 2) = Fcv(2, 2);F(2, 3) = Fcv(2, 3);
    F(3, 0) = Fcv(3, 0); F(3, 1) = Fcv(3, 1); F(3, 2) = Fcv(3, 2);F(3, 3) = Fcv(3, 3);
}

template <typename EIGEN_TYPE>
inline void copy(const  EIGEN_TYPE &Fcv, vector4d &F)
{
    F(0) = Fcv(0); F(1) = Fcv(1);
    F(2) = Fcv(2); F(3) = Fcv(3);
}

template <typename EIGEN_TYPE>
inline void copy(const EIGEN_TYPE &F, matrix3d &Fcv)
{
    Fcv(0, 0) = F(0, 0); Fcv(0, 1) = F(0, 1); Fcv(0, 2) = F(0, 2);
    Fcv(1, 0) = F(1, 0); Fcv(1, 1) = F(1, 1); Fcv(1, 2) = F(1, 2);
    Fcv(2, 0) = F(2, 0); Fcv(2, 1) = F(2, 1); Fcv(2, 2) = F(2, 2);
}


inline cv::Vec2d from_homogenus(const cv::Vec3d& p){
    return cv::Vec2d(p(0),p(1));
}

inline cv::Vec3d from_homogenus(const cv::Vec4d& p){
    return cv::Vec3d(p(0),p(1),p(2));
}

inline cv::Vec3d to_homogenus(const cv::Vec2d& p){
    return cv::Vec3d(p(0),p(1),1.);
}

inline cv::Vec4d to_homogenus(const cv::Vec3d& p){
    return cv::Vec4d(p(0),p(1),p(2), 1.);
}

inline int count_iteraion(int count_points, double w, int n){
    double T = static_cast<double>(count_points);
    double I = w * T;
    double q = 1.;
    for (int i = 0; i < n; ++i) {
        q *= (I - double(i)) / (T - double(i));
    }
    return std::log(1. - (1.-q)) / std::log(1. - q) ;

}

