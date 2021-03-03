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
//    std::cout << " PS " << Ps[0] <<std::endl;
//    std::cout << " PS monor " << Ps[0].get_minor<1,4>(2,0) <<std::endl;
    cv::Matx44d Acv;
    for (int i = 0; i < count; ++i) {

        std::cout << "f " << Ps[i].get_minor<1,4>(2,0) * ms[i](0) << std::endl;
        std::cout << "f " << Ps[i].get_minor<1,4>(0,0) << std::endl;
        std::cout << "f " << Ps[i].get_minor<1,4>(2,0) * ms[i](0) - Ps[i].get_minor<1,4>(0,0) << std::endl;
         Acv.get_minor<1,4>(i * 2,0) = Ps[i].get_minor<1,4>(2,0) * ms[i](0) - Ps[i].get_minor<1,4>(0,0);
         Acv.get_minor<1,4>(i* 2 + 1,0) = Ps[i].get_minor<1,4>(2,0) * ms[i](1) - Ps[i].get_minor<1,4>(1,0);

    }

    Eigen::Matrix4d A;
    copy(Acv, A);

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Vector4d rhs(0., 0., 0.,0.);
    Eigen::Vector4d res = svda.solve(rhs);

    cv::Vec4d out;
    copy(res, out);
    return out;


    // составление однородной системы + SVD
    // без подвохов
//    throw std::runtime_error("not implemented yet");
}
