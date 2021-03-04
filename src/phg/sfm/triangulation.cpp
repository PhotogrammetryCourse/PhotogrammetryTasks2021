#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

#include <iostream>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии,
// там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    if (count != 2) {
        throw std::runtime_error("I can triangulate only for count = 2");
    }

    //  A:
    //  x0(p0_31, p0_32, p0_33, p0_34) - (p0_11, p0_12, p0_13, p0_14)
    //  y0(p0_31, p0_32, p0_33, p0_34) - (p0_21, p0_22, p0_23, p0_24)
    //  x1(p1_31, p1_32, p1_33, p1_34) - (p1_11, p1_12, p1_13, p1_14)
    //  y1(p1_31, p1_32, p1_33, p1_34) - (p1_21, p1_22, p1_23, p1_24)

    double x0 = ms[0][0];
    double y0 = ms[0][1];
    double x1 = ms[1][0];
    double y1 = ms[1][1];

    matrix4d A;
    auto v1 = x0 * Ps[0].row(2) - Ps[0].row(0);
    auto v2 = y0 * Ps[0].row(2) - Ps[0].row(1);
    auto v3 = x1 * Ps[1].row(2) - Ps[1].row(0);
    auto v4 = y1 * Ps[1].row(2) - Ps[1].row(1);
    A << v1(0, 0) , v1(0, 1), v1(0, 2), v1(0, 3),
         v2(0, 0) , v2(0, 1), v2(0, 2), v2(0, 3),
         v3(0, 0) , v3(0, 1), v3(0, 2), v3(0, 3),
         v4(0, 0) , v4(0, 1), v4(0, 2), v4(0, 3);

    Eigen::Matrix4d Aeg;
    copy(A, Aeg);


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Aeg, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd right_null_space = svd.matrixV().col(3);
    return {right_null_space[0], right_null_space[1], right_null_space[2], right_null_space[3]};
}
