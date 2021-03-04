#include "triangulation.h"

#include "defines.h"
#include <iostream>

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    Eigen::Matrix4d A_E;
    for (int i = 0; i < count; ++i) {
        cv::Matx14d v1 = Ps[i].row(2) * ms[i](0) - Ps[i].row(0);
        cv::Matx14d v2 = Ps[i].row(2) * ms[i](1) - Ps[i].row(1);

        A_E.block<1,4>(i*2,0) << v1(0), v1(1), v1(2), v1(3);
        A_E.block<1,4>(i*2+1,0) << v2(0), v2(1), v2(2), v2(3);

    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A_E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector4d res = svda.matrixV().col(svda.matrixV().cols()-1);

    cv::Vec4d out;
    copy(res, out);
    return out;


    // составление однородной системы + SVD
    // без подвохов
//    throw std::runtime_error("not implemented yet");
}
