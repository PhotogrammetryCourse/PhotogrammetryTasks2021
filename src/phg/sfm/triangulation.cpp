#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>
#include <iostream>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
    using mat = Eigen::MatrixXd;
    mat A(count * 2, 4);

    for (int i = 0; i < count; ++i) {
        auto v1 = ms[i][0] * Ps[i].row(2) - Ps[i].row(0);
        auto v2 = ms[i][1] * Ps[i].row(2) - Ps[i].row(1);

        //std::cout << "v1:" << v1 << "\n";
        //std::cout << "v2:" << v2 << "\n";
        A.row(i*2) << v1(0), v1(1), v1(2), v1(3);
        A.row(i*2 + 1) << v2(0), v2(1), v2(2), v2(3);
    }

    //std::cout << "A:" << A << "\n";
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    int lst_col = svd.matrixV().cols() - 1;
    auto res = svd.matrixV().col(lst_col);
    //std::cout << "res" << res <<"\n";

    cv::Vec4d res2;
    res2 << res[0], res[1], res[2], res[3];

    //std::cout << "res2" << res <<"\n";
    return res2;
}
