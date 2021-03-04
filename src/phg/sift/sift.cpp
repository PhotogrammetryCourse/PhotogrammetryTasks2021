#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <libutils/rasserts.h>
#include <libutils/timer.h>

// Ссылки:
// [lowe04] - Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004
//
// Примеры реализаций (стоит обращаться только если совсем не понятны какие-то места):
// 1) https://github.com/robwhess/opensift/blob/master/src/sift.c
// 2) https://gist.github.com/lxc-xx/7088609 (адаптация кода с первой ссылки)
// 3) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.dispatch.cpp (адаптация кода с первой ссылки)
// 4) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.simd.hpp (адаптация кода с первой ссылки)

#define DEBUG_ENABLE     0
#define DEBUG_PATH       std::string("data/debug/test_sift/debug/")
#define INCREMENTAL_SIGMA 1

#define NOCTAVES                    3                    // число октав
#define OCTAVE_NLAYERS              3                    // в [lowe04] это число промежуточных степеней размытия картинки в рамках одной октавы обозначается - s, т.е. s слоев в каждой октаве
#define OCTAVE_GAUSSIAN_IMAGES      (OCTAVE_NLAYERS + 3)
#define OCTAVE_DOG_IMAGES           (OCTAVE_NLAYERS + 2)
#define INITIAL_IMG_SIGMA           0.75                 // предполагаемая степень размытия изначальной картинки
#define INPUT_IMG_PRE_BLUR_SIGMA    1.0                  // сглаживание изначальной картинки
#define SIGMA_SCALE                 1.5

#define SUBPIXEL_FITTING_ENABLE      0    // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить

#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW           16    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.8 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно
#define MAX_POINTS_FROM_ANGLES 3

#define DESCRIPTOR_WINDOW 16 // 16x16 для дескриптора точки
#define DESCRIPTOR_ANGLE_STEP 45
#define KP_DESCRIPTOR_SIZE (DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS)

#define PI 3.14159265

namespace {
    timer t;
    std::string log_prefix = "[MY SIFT TIMING] ";

    // https://stackoverflow.com/a/29730828
    cv::Point2f rotate2d(const cv::Point2f& inPoint, const float& angRad)
    {
        cv::Point2f outPoint;
        //CW rotation
        outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
        outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;
        return outPoint;
    }

    cv::Point2f rotatePoint(const cv::Point2f& inPoint, const cv::Point2f& center, const float& angDeg)
    {
        return rotate2d(inPoint - center, angDeg * PI / 180.0) + center;
    }

    // https://stackoverflow.com/a/61395220
    cv::Mat getKernel(int size, float sigma) {
        cv::Mat kernel = cv::Mat::zeros(size, size, CV_32FC1);
        int mid = kernel.rows / 2;
        kernel.at<float>(mid, mid) = 1;
        cv::GaussianBlur(kernel, kernel, {0, 0}, sigma);
        return kernel;
    }

    float parabolaFitting(float x0, float x1, float x2) {
        rassert((x1 >= x0 && x1 >= x2) || (x1 <= x0 && x1 <= x2), 12541241241241);
        float a = (x2-2.0f*x1+x0) / 2.0f;
        float b = x1 - x0 - a;
        //  так как интересно смещение относительно 1
        float shift = - b / (2.0f * a) - 1.0f;
        return shift;
    }

    void showImg(const std::string &text, cv::Mat &img) {
        cv::Mat normalized;
        cv::normalize(img, normalized, 255, 0, cv::NORM_MINMAX);
        cv::imshow(text, normalized);
        cv::waitKey();
    }

    // https://dsp.stackexchange.com/a/3386
    void approximate(cv::Point3f &delta, float &contrast, const cv::Point2i &cds,
                     const cv::Mat &below, const cv::Mat &current, const cv::Mat &above) {
        // -df(x0)
        cv::Mat grad(3, 1, CV_32FC1);
        grad.at<float>(0, 0) = -(current.at<float>(cds.x + 1, cds.y) - current.at<float>(cds.x - 1, cds.y)) * 0.5f;
        grad.at<float>(1, 0) = -(current.at<float>(cds.x, cds.y + 1) - current.at<float>(cds.x, cds.y - 1)) * 0.5f;
        grad.at<float>(2, 0) = -(above.at<float>(cds.x, cds.y) - below.at<float>(cds.x, cds.y)) * 0.5f;

        /*
         * xx xy xz
         * yx yy yz
         * zx zy zz
         */

        cv::Mat hessian = cv::Mat(3, 3, CV_32FC1);
        hessian.at<float>(0, 0) = current.at<float>(cds.x + 1, cds.y) +
                                  current.at<float>(cds.x - 1, cds.y) -
                                  2.0f * current.at<float>(cds.x, cds.y);

        hessian.at<float>(1, 1) = current.at<float>(cds.x, cds.y + 1) +
                                  current.at<float>(cds.x, cds.y - 1) -
                                  2.0f * current.at<float>(cds.x, cds.y);

        hessian.at<float>(2, 2) = above.at<float>(cds.x, cds.y) +
                                  below.at<float>(cds.x, cds.y) -
                                  2.0f * current.at<float>(cds.x, cds.y);
        // суммируем угловые точки в плоскости xy
        hessian.at<float>(0, 1) = (current.at<float>(cds.x + 1, cds.y + 1) +
                                    current.at<float>(cds.x - 1, cds.y - 1) +
                                    current.at<float>(cds.x + 1, cds.y - 1) +
                                    current.at<float>(cds.x - 1, cds.y + 1)) * 0.25f;
        // суммируем угловые точки в плоскости xz
        hessian.at<float>(0, 2) = (above.at<float>(cds.x - 1, cds.y) +
                                   below.at<float>(cds.x + 1, cds.y) +
                                   above.at<float>(cds.x + 1, cds.y) +
                                   below.at<float>(cds.x - 1, cds.y)) * 0.25f;
        // суммируем угловые точки в плоскости yz
        hessian.at<float>(1, 2) = (above.at<float>(cds.x, cds.y + 1) +
                                   below.at<float>(cds.x, cds.y - 1) +
                                   above.at<float>(cds.x, cds.y - 1) +
                                   below.at<float>(cds.x, cds.y + 1)) * 0.25f;


        hessian.at<float>(1, 0) = hessian.at<float>(0, 1);
        hessian.at<float>(2, 0) = hessian.at<float>(0, 2);
        hessian.at<float>(2, 1) = hessian.at<float>(1, 2);


        cv::Mat res;
        cv::solve(hessian, grad, res);

        delta.x = res.at<float>(0, 0);
        delta.y = res.at<float>(1, 0);
        delta.z = res.at<float>(2, 0);

        contrast = current.at<float>(cds.x, cds.y) +
                0.5f * -(grad.at<float>(0, 0) * delta.x + grad.at<float>(1, 0) * delta.y + grad.at<float>(2, 0) * delta.z);
    }

    bool harris(const cv::Point2i &cds, const cv::Mat &current) {
        static float r = 10.0;
        static float threshold = (r + 1) * (r + 1) / r;
        float dxx = current.at<float>(cds.x + 1, cds.y) +
                    current.at<float>(cds.x - 1, cds.y) -
                    2.0f * current.at<float>(cds.x, cds.y);
        float dyy = current.at<float>(cds.x, cds.y + 1) +
                    current.at<float>(cds.x, cds.y - 1) -
                    2.0f * current.at<float>(cds.x, cds.y);
        float dxy = (current.at<float>(cds.x + 1, cds.y + 1) +
                     current.at<float>(cds.x - 1, cds.y - 1) +
                     current.at<float>(cds.x + 1, cds.y - 1) +
                     current.at<float>(cds.x - 1, cds.y + 1)) * 0.25f;

        float traceH = dxx + dyy;
        float detH = dxx * dyy - dxy * dxy;

        if ((traceH * traceH / detH) < threshold) return true;
        return false;
    }
}


void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {

    cv::Mat img = originalImg.clone();

    // для удобства используем черно-белую картинку и работаем с вещественными числами (это еще и может улучшить точность)
    if (originalImg.type() == CV_8UC1) { // greyscale image
        img.convertTo(img, CV_32FC1, 1.0);
    } else if (originalImg.type() == CV_8UC3) { // BGR image
        img.convertTo(img, CV_32FC3, 1.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        rassert(false, 14291409120);
    }

    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "01_grey.png", img);
    cv::GaussianBlur(img, img, {0, 0}, INPUT_IMG_PRE_BLUR_SIGMA, INPUT_IMG_PRE_BLUR_SIGMA);
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "02_grey_blurred.png", img);

//
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> DoGPyramid;
    buildPyramids(img, gaussianPyramid, DoGPyramid);

    findLocalExtremasAndDescribe(gaussianPyramid, DoGPyramid, kps, desc);
}


void phg::SIFT::buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid,
                              std::vector<cv::Mat> &DoGPyramid) {

    cv::Mat layerBase = imgOrg;
    double step = pow(2.0, 1.0 / OCTAVE_NLAYERS);

    double currentSigma = INITIAL_IMG_SIGMA;
    if (DEBUG_ENABLE) t.restart();

    for (int octave = 0; octave < NOCTAVES; ++octave) {
        gaussianPyramid.push_back(layerBase);

        // заполним октаву гауссиан
        for (int layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            cv::Mat blurred;
#ifdef INCREMENTAL_SIGMA
            currentSigma *= step;
            cv::GaussianBlur(layerBase, blurred, {0, 0}, currentSigma, currentSigma);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "03_g_" + to_string(octave) + "_i_" + to_string(layer) + "0.png", blurred);
#else
            double sigmaPrev = INITIAL_IMG_SIGMA * pow(step, layer - 1); // sigma1  - сигма до которой дошла картинка на предыдущем слое
            double sigmaCur  = INITIAL_IMG_SIGMA  * pow(step, layer);     // sigma12 - сигма до которой мы хотим дойти на текущем слое
            currentSigma = sqrt(sigmaCur*sigmaCur - sigmaPrev*sigmaPrev);
            cv::GaussianBlur(gaussianPyramid.back(), blurred, {0, 0}, currentSigma, currentSigma);

            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "03_g_" + to_string(octave) + "_i_" + to_string(layer) + "1.png", blurred);
#endif
            gaussianPyramid.push_back(blurred);
        }

        // теперь октаву лаплассиан
        for (int layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            uint32_t blurredIdx = OCTAVE_GAUSSIAN_IMAGES * octave + layer;
            cv::Mat dog;
            cv::subtract(gaussianPyramid[blurredIdx + 1], gaussianPyramid[blurredIdx], dog);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "04_dog_" + to_string(octave) + "_i_" + to_string(layer) + ".png", dog);
            DoGPyramid.push_back(dog);
        }

        // уменьшим картинку
        cv::Mat nextLayerBase;
        cv::resize(gaussianPyramid.back(), nextLayerBase, {}, 0.5, 0.5, cv::INTER_NEAREST);
        layerBase = nextLayerBase;
#ifdef INCREMENTAL_SIGMA
        currentSigma = INITIAL_IMG_SIGMA;
#endif
    }

    if (DEBUG_ENABLE) std::cout << log_prefix << "buildPyramids: " << t.elapsed() << std::endl;
}


cv::Mat phg::SIFT::buildDescriptor(const cv::Mat &gaussPic, const cv::Point2i &pix, const cv::KeyPoint &keyPoint
                        , int layer ) {
    int current_d_size;
    int current_w_size;
    if (layer == 1) {
        current_d_size = 2;
    } else if (layer == 2) {
        current_d_size = 3;
    } else if (layer == 3) {
        current_d_size = 4;
    }
    current_w_size = DESCRIPTOR_SAMPLES_N * current_d_size;

    cv::Mat descriptorKernel =  getKernel(current_w_size + 1, keyPoint.size / 2);
    cv::Mat descriptor = cv::Mat::zeros(1, KP_DESCRIPTOR_SIZE, CV_32FC1);
    // пусть x  будет на позиции (8, 8)
    int delta = descriptorKernel.rows / 2;
    int shiftX = std::max(pix.x - delta, 0), shiftY = std::max(pix.y - delta, 0);

    for (int descX = 0; descX < DESCRIPTOR_SAMPLES_N; ++descX) {
        for (int descY = 0; descY < DESCRIPTOR_SAMPLES_N; ++descY) {
            auto bin = descriptor.begin<float>() + DESCRIPTOR_NBINS * (descX * DESCRIPTOR_SAMPLES_N + descY);
            for (int x = 0; x < current_d_size; ++x) {
                for (int y = 0; y < current_d_size; ++y) {
                    int xg = current_d_size * descX + x;
                    int yg = current_d_size * descY + y;

                    cv::Point2i ptTransformed =
                            rotatePoint({(float)(xg + shiftX), (float)(yg + shiftY)}, pix, keyPoint.angle);

                    int xs = ptTransformed.x;
                    int ys = ptTransformed.y;
                    if (xs <= 0 || xs >= gaussPic.rows - 1 | ys <= 0 || ys >= gaussPic.cols - 1) continue;
                    float dx = gaussPic.at<float>(xs + 1, ys) - gaussPic.at<float>(xs - 1, ys);
                    float dy = gaussPic.at<float>(xs, ys + 1) - gaussPic.at<float>(xs, ys - 1);
                    float m = std::sqrt(std::pow(dx, 2.0f) + std::pow(dy, 2.0f));
                    float th = std::atan2(dy, dx)  * 180.0 / PI;
                    if (isnanf(th)) continue;
                    if (th < 0) th += 360;
                    th -= keyPoint.angle;
                    if (th < 0) th += 360;
                    if (!(th < 360 && th >= 0)) {
                        continue;
                    }
                    * (bin + (int) (th / DESCRIPTOR_ANGLE_STEP)) += m * descriptorKernel.at<float>(xg, yg);
                }
            }
        }
    }

    cv::normalize(descriptor, descriptor);
    return descriptor;
}

void phg::SIFT::getPointOrientationAndDescriptor(const cv::Mat &gaussPic, const cv::Point2i &pix, int layer,
                                                 const cv::Point2f &point, float sigma, float contrast,
                                                 std::vector<cv::KeyPoint> &points, std::vector<cv::Mat> &descriptors) {
    int current_w_r;
    if (layer == 1) {
        current_w_r = 8;
    } else if (layer == 2) {
        current_w_r = 12;
    } else if (layer == 3) {
        current_w_r = 16;
    }
    cv::Mat kernel = getKernel(2*current_w_r + 1, current_w_r * sigma / 2);

    int delta = kernel.rows / 2;
    int shiftX = std::max(pix.x - delta, 0), shiftY = std::max(pix.y - delta, 0);
    int xStart = shiftX == 0 ? 1 : 0, xEnd = std::min(pix.x + delta, gaussPic.rows - 1);
    int yStart = shiftY == 0 ? 1 : 0, yEnd = std::min(pix.y + delta, gaussPic.cols - 1);

    float bins[ORIENTATION_NHISTS]{0.0};

    for (int x = xStart; x + shiftX < xEnd; ++x) {
        for (int y = yStart; y + shiftY < yEnd; ++y) {
            int xs = x + shiftX, ys = y + shiftY;
            float dx = gaussPic.at<float>(xs + 1, ys) - gaussPic.at<float>(xs - 1, ys);
            float dy = gaussPic.at<float>(xs, ys + 1) - gaussPic.at<float>(xs, ys - 1);
            float m = std::sqrt(std::pow(dx, 2.0f) + std::pow(dy, 2.0f));
            float th = std::atan2(dy, dx)  * 180.0 / PI;
            if (th < 0) th += 360.0;
            bins[(int) (th / 10.0)] += m * kernel.at<float>(x, y);
        }
    }

    float max = *std::max_element(bins, bins + ORIENTATION_NHISTS);
    int counter = 0;
    for (int i = 0; i < ORIENTATION_NHISTS; ++i) {
        // берем только пики
        if (bins[i] >= max * ORIENTATION_VOTES_PEAK_RATIO) {
            int x0Pos = i == 0 ? ORIENTATION_NHISTS - 1 : i - 1;
            int x2Pos = i == ORIENTATION_NHISTS - 1 ? 0 : i + 1;
            // если не пик, то выбрасываем
            if (!(bins[i] >= bins[x0Pos] && bins[i] >= bins[x2Pos])) continue;
            // Пусть для градусов будем брать серидину интервала
            float deg0 = 10.0f * (x0Pos + 0.5f);
            float deg1 = 10.0f * (i + 0.5f);
            float deg2 = 10.0f * (x2Pos + 0.5f) ;
            if (i == ORIENTATION_NHISTS - 1) {
                deg2 += 360.0f;
            } else if (i == 0) {
                deg1 += 360.0f;
                deg2 += 360.0f;
            }
            float shift = parabolaFitting(bins[x0Pos], bins[i], bins[x2Pos]);
            float angle = shift < 0 ? -shift * deg0 + (1 + shift) * deg1 : (1 - shift) * deg1 + shift * deg2;
            if (angle >= 360) angle -= 360;
            counter ++;

            cv::KeyPoint kp{point, 10 * sigma, angle, contrast};
            points.push_back(kp);

            descriptors.push_back(buildDescriptor(gaussPic, pix, kp,  layer));
        }
        if (counter >= MAX_POINTS_FROM_ANGLES) break;
    }

}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid,
                                             const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) const {
    std::vector<cv::Mat> descriptors;
    if (DEBUG_ENABLE) t.restart();
    for (int octave = 0; octave < NOCTAVES; ++octave) {
        float octaveScale = std::pow(2.0, octave);

        for (int layer = 1; layer < OCTAVE_NLAYERS + 1; ++layer) {

            cv::Mat below = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1];
            cv::Mat current = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
            cv::Mat above = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1];

            for (int x = 0; x < current.rows; ++x) {
                for (int y = 0; y < current.cols; ++y) {

                    // границы
                    if (x <= 0 || y <= 0 || x >= current.rows - 1 || y >= current.cols - 1) continue;
                    float pixelVal = std::fabs(current.at<float>(x, y));
                    // фильтр соседей
                    bool good = true;
                    for (int lx = x - 1; lx <= x + 1; lx ++) {
                        for (int ly = y - 1; ly <= y + 1; ly++) {
                            if (pixelVal < std::fabs(below.at<float>(lx, ly)) ||
                                pixelVal < std::fabs(above.at<float>(lx, ly)) ||
                                pixelVal < std::fabs(current.at<float>(lx, ly))) {
                                good = false;
                                break;
                            }
                        }
                    }
                    if (!good) continue;
                    cv::Point2i pixel(x, y);
                    cv::Point3f delta{};
                    float contrast;

                    approximate(delta, contrast, pixel, below, current, above);
                    // просто отбросим нестабильную точку
                    if (std::fabs(delta.x) > 0.5f || std::fabs(delta.y) > 0.5f || std::fabs(delta.z) > 0.5f) continue;

                    // >> почему порог контрастности должен уменьшаться при увеличении числа слоев в октаве?
                    // потому что потенциально у нас теперь больше ключевых точеки мы можем ужесточить отбор?
                    if (abs(contrast) < contrast_threshold / OCTAVE_NLAYERS) continue;

                    // добавить проверку с изгибом, что рассматриваем не линию
                    if (!harris(pixel, current)) continue;

                    double step = pow(2.0, 1.0 / OCTAVE_NLAYERS);
                    float currentSigma = INITIAL_IMG_SIGMA * pow(step, layer + 1);
                    float prevSigma = INITIAL_IMG_SIGMA * pow(step, layer);
                    float nextSigma = INITIAL_IMG_SIGMA * pow(step, layer + 2);
                    currentSigma = delta.z < 0 ?  -delta.z * prevSigma + (1 + delta.z) * currentSigma :
                                   (1 - delta.z) * currentSigma + delta.z * nextSigma;


                    cv::Mat gP = gaussianPyramid[OCTAVE_GAUSSIAN_IMAGES * octave + layer + 1];
                    cv::Point2f point{(0.5f + pixel.y + delta.y) * octaveScale, (0.5f + pixel.x + delta.x) * octaveScale};

                    getPointOrientationAndDescriptor(gP, pixel, layer, point, currentSigma, abs(contrast), keyPoints,
                                                     descriptors);
                }
            }
        }
    }
    if (DEBUG_ENABLE) std::cout << log_prefix << "findLocalExtremasAndDescribe: count time: " << t.elapsed() << std::endl;
    if (DEBUG_ENABLE) t.restart();
    desc = cv::Mat(descriptors.size(), KP_DESCRIPTOR_SIZE, CV_32FC1);
    for (int i = 0; i < desc.rows; ++i) {
        descriptors[i].copyTo(desc.row(i));
    }
    if (DEBUG_ENABLE) std::cout << log_prefix << "findLocalExtremasAndDescribe: merge time: " << t.elapsed() << std::endl;
}