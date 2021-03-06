#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    const size_t a_rows = 2 * count;
    const size_t a_cols = 4;

    const size_t min_sv_idx = std::min(a_rows, a_cols) - 1;
        
    Eigen::MatrixXd A(a_rows, a_cols);

    for (size_t i = 0; i < count; ++i) {
        const auto row0 = ms[i][0] * Ps[i].row(2) - Ps[i].row(0);
        const auto row1 = ms[i][1] * Ps[i].row(2) - Ps[i].row(1);

        A.row(2 * i    ) << row0(0, 0), row0(0, 1), row0(0, 2), row0(0, 3);
        A.row(2 * i + 1) << row1(0, 0), row1(0, 1), row1(0, 2), row1(0, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd null_space = svda.matrixV().col(min_sv_idx);

    return cv::Vec4d(null_space[0], null_space[1], null_space[2], null_space[3]);
}
