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
    const double x = point[0] / point[2];
    const double y = point[1] / point[2];

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double l = (1.0 + k1_ * r2 + k2_ * r4);

    return cv::Vec3d(
        x * l * f_ + cx_ + width_ * 0.5, 
        y * l * f_ + cy_ + height_ * 0.5, 
        1.0
    );
}

cv::Vec3d phg::Calibration::unproject(const cv::Vec2d &pixel) const
{
    const double x = (pixel[0] - cx_ - width_ * 0.5) / f_;
    const double y = (pixel[1] - cy_ - height_ * 0.5) / f_;

    const double r2 = x * x + y * y;
    const double r4 = r2 * r2;
    const double l = 1.0 + k1_ * r2 + k2_ * r4;

    return cv::Vec3d(x / l, y / l, 1.0);
}
