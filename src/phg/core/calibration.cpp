#include <phg/sfm/defines.h>
#include "calibration.h"

#include <iostream>

// #define ITERATIVE_UNPROJECT

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
    return {
        f_, 0., cx_ + width_ * 0.5, 
        0., f_, cy_ + height_ * 0.5, 
        0., 0., 1.
    };
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

    const double r2 = x * x + y * y;
    const double l = 1 + (k1_ + k2_ * r2) * r2;

    x *= l;
    y *= l;

    x += cx_ + width_ * 0.5;
    y += cy_ + height_ * 0.5;

    return cv::Vec3d(x, y, 1.0);
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    double x0 = (pixel[0] - cx_ - width_ * 0.5) / f_;
    double y0 = (pixel[1] - cy_ - height_ * 0.5) / f_;

    // TODO 12: добавьте учет радиальных искажений, когда реализуете - подумайте: почему строго говоря это - не симметричная формула формуле из project? (но лишь приближение)
    // Минутка честных расписываний :D
    // (1) x - xc == (2 + k1 * r2 + k2 * r2^2) * dx
    // (2) y - yc == (2 + k1 * r2 + k2 * r2^2) * dy
    // r2 := dx^2 + dy^2

    // q := (x - xc) / (y - yc)
    // разделим (1) на (2)
    // => dx = q * dy
    
    // подставим в определение r2
    // p := 1 + q^2
    // => r2 = p * dy^2

    // подставим всё в (1)
    // y - yc == (2 + k1 * p * dy^2 + k2 * p^2 * dy^4) * dy
    // еее, полином 5 степени))

    // лучше итеративно http://peterabeles.com/blog/?p=73
    static const double EPS = 1e-3;

#ifdef ITERATIVE_UNPROJECT
    double x = x0;
    double y = y0;
    double m = 0;
    while (true) {
        const double r2 = x * x + y * y;
        m = (k1_ + k2_ * r2) * r2;

        if (m < EPS)
            break;

        x = x0 / (1 + m);
        y = y0 / (1 + m);
    }

    const auto unprojected = cv::Vec3d(
        (x + cx_ * m) / (1 + m), 
        (y + cy_ * m) / (1 + m), 
        1.0
    );
#else
    const double r2 = x0 * x0 + y0 * y0;
    const double l = 1 + (k1_ + k2_ * r2) * r2;

    const auto unprojected = cv::Vec3d(
        x0 / l, 
        y0 / l, 
        1.0
    );
#endif

    if (false) {
        const auto reprojected = project(unprojected);
        
        std::cout << "unprojection error: " << std::sqrt(std::pow(reprojected[0] - pixel[0], 2) + std::pow(reprojected[1] - pixel[1], 2)) << "\n";
    }

    return unprojected;
}
