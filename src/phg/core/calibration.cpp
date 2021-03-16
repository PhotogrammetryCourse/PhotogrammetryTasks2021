#include <phg/sfm/defines.h>
#include "calibration.h"

#include "iostream"

namespace newton{

inline double f_x(double x, double k1, double k2, double n){
    double x2 = x* x;
    return x * (1 + k1 *x2 + x2*x2* k2) - n;
}
double found_extreme(double init_value, double k1, double k2, double n){

    double delta = 0.3;
    double fx = f_x(init_value, k1,k2, n);
    double fx1 = f_x(init_value - delta, k1,k2, n);
    double fx2 = f_x(init_value - 2 * delta, k1,k2, n);



    double x = init_value;
    double x_n = 0.;

    double x_prev2 = x - delta;
    double x_prev3 = x - 3 * delta;

    int max_step = 30000;
    int step = 0;

    do{
        double dyx = (fx - fx1) / (x - x_prev2);
        double dyx2 = (fx1 - fx2) / (x_prev2 - x_prev3);
        double dyxx = (dyx - dyx2) /((x -  x_prev3));

        x_n = x - dyx / dyxx;

        if(x_n < 0)
            break;

        x_prev3 = x_prev2;
        x_prev2 = x;
        x = x_n;
        fx2 = fx1;
        fx1 = fx;
        fx = f_x(x_n, k1,k2,n);
        step+=1;
    }
    while(std::fabs(fx) > 1);

    return x_n;
}
}

phg::Calibration::Calibration(int width, int height)
    : width_(width)
    , height_(height)
    , cx_(0)
    , cy_(0)
    , k1_(0)
    , k2_(0)
{
    // 50mm guess

    double diag_35mm = 36.0 * 36.0 + 24.0 * 24.0;
    double diag_pix = (double) width * (double) width + (double) height * (double) height;

    f_ = 50.0 * std::sqrt(diag_pix / diag_35mm);
}

cv::Matx33d phg::Calibration::K() const {
    return {f_, 0., cx_ + width_ * 0.5, 0., f_, cy_ + height_ * 0.5, 0., 0., 1.};
}

int phg::Calibration::width() const {
    return width_;
}

int phg::Calibration::height() const {
    return height_;
}

cv::Vec3d phg::Calibration::project(const cv::Vec3d &point) const
{
    double x = f_ * point[0] / point[2];
    double y = f_ * point[1] / point[2];

    double r = x * x + y * y;
    double distortion = 1 + k1_ * r + k2_*r*r;

    x *= distortion;
    y *= distortion;

    // TODO 11: добавьте учет радиальных искажений (k1_, k2_)

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    // TODO 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)
    if(std::fabs(k1_) > 0)
    {
        double R = x * x + y * y;

        /*
        Я решил искать значение радиуса численными методом. Возможно не всегда он правильно находит, потому что проверка на количество инлайнеров не проходят 
*/
        double r = newton::found_extreme(std::sqrt(R), k1_,k2_, std::sqrt(R));
        r = std::fabs(r) < 1 ? R : r;
        double dist = 1. + k1_ * r + r * r * k2_;

        x = x / dist;
        y = y / dist;

    }

    x /= f_;
    y /= f_;

    return cv::Vec3d(x, y, 1.0);
}
