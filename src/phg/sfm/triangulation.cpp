#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    // TODO - done
    Eigen::MatrixXd A;
    A.resize(2 * count, 4);
    for (int i = 0; i < count; i++) {
        cv::Matx14d r0 = ms[i][0] * Ps[i].row(2) - ms[i][2] * Ps[i].row(0);
        cv::Matx14d r1 = ms[i][1] * Ps[i].row(2) - ms[i][2] * Ps[i].row(1);
        A.row(2 * i) << r0(0), r0(1), r0(2), r0(3);
        A.row(2 * i + 1) << r1(0), r1(1), r1(2), r1(3);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto result = svda.matrixV().col(svda.matrixV().cols() - 1);
    return cv::Vec4d(result.x(), result.y(), result.z(), result.w());
}
