#include <phg/sfm/defines.h>
#include "calibration.h"


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

    // TODO 11: добавьте учет радиальных искажений (k1_, k2_)

    double r2 = x * x + y * y;
    double r4 = r2 * r2;

    x *= 1 + k1_ * r2 + k2_ * r4;
    y *= 1 + k1_ * r2 + k2_ * r4;
    // TODO height_ and width_ тут не нравятся ...
    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double x = pixel[0] - cx_ - width_ * 0.5;
    double y = pixel[1] - cy_ - height_ * 0.5;

    // TODO 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)
    // есть ещё такой вариант, но сами авторы пишут, что ошибка высокая
    // https://www.researchgate.net/publication/224599781_Lens_Distortion_Correction_Using_Ideal_Image_Coordinates

    // Это не тот же самый R^2, что был. Если у нас "выпуклая" дисторсия, то таким "обратным"
    // преобразованием мы "недоведем" точку до того места, где она была.
    //  Если это была "вогнутая" дисторсия, то в результате останется немного выпуклая.
    double r2_d = x * x + y * y;
    double r4_d = r2_d * r2_d;

    x /= 1 + k1_ * r2_d + k2_ * r4_d;
    y /= 1 + k1_ * r2_d + k2_ * r4_d;

    x /= f_;
    y /= f_;

    return cv::Vec3d(x, y, 1.0);
}
