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

    int a_rows = 2 * count;
    int a_cols = 4;

    Eigen::MatrixXd A(a_rows, a_cols);

    for (int i = 0; i < count; ++i) {
        cv::Matx14d a0 = Ps[i].row(2) * ms[i](0) - Ps[i].row(0) * ms[i](2);
        cv::Matx14d a1 = Ps[i].row(2) * ms[i](1) - Ps[i].row(1) * ms[i](2);

        A.row(2 * i) << a0(0), a0(1), a0(2), a0(3);
        A.row(2 * i + 1) << a1(0), a1(1), a1(2), a1(3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd null_space = svda.matrixV().col(a_cols - 1);

    return cv::Vec4d(null_space[0], null_space[1], null_space[2], null_space[3]);
}
