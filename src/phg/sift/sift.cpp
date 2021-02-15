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

#define DEBUG_ENABLE     1
#define DEBUG_PATH       std::string("data/debug/test_sift/debug/")

#define NOCTAVES                    3                    // число октав
#define OCTAVE_NLAYERS              3                    // в [lowe04] это число промежуточных степеней размытия картинки в рамках одной октавы обозначается - s, т.е. s слоев в каждой октаве
#define OCTAVE_GAUSSIAN_IMAGES      (OCTAVE_NLAYERS + 3)
#define OCTAVE_DOG_IMAGES           (OCTAVE_NLAYERS + 2)
#define INITIAL_IMG_SIGMA           0.75                 // предполагаемая степень размытия изначальной картинки
#define INPUT_IMG_PRE_BLUR_SIGMA    1.0                  // сглаживание изначальной картинки
   
#define SUBPIXEL_FITTING_MAX_STEPS 5      // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить

#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE              4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS             8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_INTERPOLATE_HISTS 1 // интерполировать ли гистограммы
#define DESCRIPTOR_SAMPLES_N         4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R   1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно

#define USE_BASE_IN_OCTAVE_GENERATION 1
#define CURVATURE_THRESHOLD           0.05 // дополнительный фильтр по кривизне функции -- позволяет отсечь выраженные границы

// кажется, ничего особо не меняется (по крайней мере, в чиселках из вывода test_sift)
#define ORIENTATION_HISTOGRAM_SMOOTH_DISTANCE 3 

struct run_timed {
    std::string description;
    timer t;

    run_timed(std::string description) 
    : description(std::move(description)) {
    }

    ~run_timed() {
        if (DEBUG_ENABLE) 
            std::cerr << "Timed step [" << description << "] completed in " << t.elapsed() << "\n";
    }
};

inline size_t layer_idx(size_t octave, size_t layer) {
    return octave * OCTAVE_GAUSSIAN_IMAGES + layer;
}

inline double fix_angle(double angle) {
    while (angle < 0.0) 
        angle += 360.0;       

    while (angle >= 360.0) 
        angle -= 360.0;

    return angle;
}

inline double atan2deg(double y, double x) {
    return fix_angle(atan2(y, x) * 180.0 / M_PI + 90.0);
}

void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    // используйте дебаг в файлы как можно больше, это очень удобно и потраченное время окупается крайне сильно,
    // ведь пролистывать через окошки показывающие картинки долго, и по ним нельзя проматывать назад, а по файлам - можно
    // вы можете запустить алгоритм, сгенерировать десятки картинок со всеми промежуточными визуализациями и после запуска
    // посмотреть на те этапы к которым у вас вопросы или про которые у вас опасения
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "00_input.png", originalImg);

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
    cv::GaussianBlur(img, img, cv::Size(0, 0), INPUT_IMG_PRE_BLUR_SIGMA, INPUT_IMG_PRE_BLUR_SIGMA);
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "02_grey_blurred.png", img);

    // Scale-space extrema detection
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> DoGPyramid;
    buildPyramids(img, gaussianPyramid, DoGPyramid);

    findLocalExtremasAndDescribe(gaussianPyramid, DoGPyramid, kps, desc);
}

void phg::SIFT::buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid) {
    gaussianPyramid.resize(NOCTAVES * OCTAVE_GAUSSIAN_IMAGES);

    const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS

    // строим пирамиду гауссовых размытий картинки
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        if (octave == 0) {
            int layer = 0;
            gaussianPyramid[layer_idx(octave, layer)] = imgOrg.clone();
        } else {
            int layer = 0;
            size_t prevOctave = octave - 1;
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия (сигмы должны совпадать)
            cv::Mat img = gaussianPyramid[octave - 1].clone();
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет не идеально 2 пикселя в один схлопываться - а слегка смещаться
            cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
            gaussianPyramid[layer_idx(octave, layer)] = std::move(img);
        }

        #if USE_BASE_IN_OCTAVE_GENERATION 
        #pragma omp parallel for
        #endif
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            
            #if USE_BASE_IN_OCTAVE_GENERATION
            const auto prevLayer = 0;         // always base
            #else
            const auto prevLayer = layer - 1;
            #endif
            
            auto imgLayer = gaussianPyramid[layer_idx(octave, prevLayer)].clone();

            // если есть два последовательных гауссовых размытия с sigma1 и sigma2, то результат будет с sigma12=sqrt(sigma1^2 + sigma2^2) => sigma2=sqrt(sigma12^2-sigma1^2)
            const double sigma = INITIAL_IMG_SIGMA * sqrt(pow(k, layer << 1) - pow(k, prevLayer << 1));

            const cv::Size automaticKernelSize = cv::Size(0, 0);

            cv::GaussianBlur(imgLayer, imgLayer, automaticKernelSize, sigma, 0);

            gaussianPyramid[layer_idx(octave, layer)] = std::move(imgLayer);
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double effectiveSigma = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);

            #if USE_BASE_IN_OCTAVE_GENERATION
            const auto suffix = "_from_base.png";
            #else
            const auto suffix = "_from_prev.png";
            #endif
            
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(effectiveSigma) + suffix, gaussianPyramid[layer_idx(octave, layer)]);
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);

    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG), т.к. вычитать надо из слоя слой в рамках одной и той же октавы - то есть приятный параллелизм на уровне октав
    {
        run_timed step("building DoG");

        #pragma omp parallel for
        for (ptrdiff_t octave = 0; octave < NOCTAVES; ++octave) {
            for (size_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
                const int prevLayer = layer - 1;
                const int dogLayer = layer - 1;
                // это же матрицы, поэтому их можно сразу вычитать друг из друга + наверняка подключится подкапотная магия и будет ещё быстрее, чем с циклом
                DoGPyramid[octave * OCTAVE_DOG_IMAGES + dogLayer] = 
                    gaussianPyramid[layer_idx(octave, layer)] - 
                    gaussianPyramid[layer_idx(octave, prevLayer)];
            }
        }
    }

    // нам нужны padding-картинки по краям октавы, чтобы извлекать экстремумы, но в статье предлагается не s+2 а s+3: [lowe04] We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema detection covers a complete octave
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramidDoG/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer]);
        }
    }
}

namespace {
    float parabolaFitting(float x0, float x1, float x2) {
        rassert((x1 >= x0 && x1 >= x2) || (x1 <= x0 && x1 <= x2), 12541241241241);

        // a*0^2+b*0+c=x0
        // a*1^2+b*1+c=x1
        // a*2^2+b*2+c=x2

        // c=x0
        // a+b+x0=x1     (2)
        // 4*a+2*b+x0=x2 (3)

        // (3)-2*(2): 2*a-y0=y2-2*y1; a=(y2-2*y1+y0)/2
        // (2):       b=y1-y0-a
        float a = (x2-2.0f*x1+x0) / 2.0f;
        float b = x1 - x0 - a;
        // extremum is at -b/(2*a), but our system coordinate start (i) is at 1, so minus 1
        float shift = - b / (2.0f * a) - 1.0f;
        return shift;
    }
}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) {
    std::vector<std::vector<float>> pointsDesc;

    // 3.1 Local extrema detection
    #pragma omp parallel // запустили каждый вычислительный поток процессора
    {
        // каждый поток будет складировать свои точки в свой личный вектор (чтобы не было гонок и не были нужны точки синхронизации)
        std::vector<cv::KeyPoint> thread_points;
        std::vector<std::vector<float>> thread_descriptors;

        for (size_t octave = 0; octave < NOCTAVES; ++octave) {
            double octave_downscale = pow(2.0, octave);
            for (size_t layer = 1; layer + 1 < OCTAVE_DOG_IMAGES; ++layer) {
                const cv::Mat DoGs[3] = {
                    DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1],
                    DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer    ],
                    DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1]
                };

                // теперь каждый поток обработает свой кусок картинки 
                #pragma omp for
                for (ptrdiff_t j = 1; j < DoGs[1].rows - 1; ++j) {
                    for (ptrdiff_t i = 1; i + 1 < DoGs[1].cols; ++i) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = DoGs[1].at<float>(j, i);
                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                            for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                                for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                                    if (dz == 0 && dy == 0 && dx == 0)
                                        continue;
                                    
                                    is_max &= DoGs[1 + dz].at<float>(j + dy, i + dx) < center;
                                    is_min &= DoGs[1 + dz].at<float>(j + dy, i + dx) > center;
                                }
                            }
                        }

                        bool is_extremum = (is_min || is_max);
                        if (!is_extremum)
                            continue; // очередной элемент cascade filtering, если не экстремум - сразу заканчиваем обработку этого пикселя

                        // 4 Accurate keypoint localization
#if SUBPIXEL_FITTING_MAX_STEPS // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить
                        cv::Vec3f delta { 0, 0, 0 };
                        {
                            int dy = 0;
                            int dx = 0;
                            bool done = false;
                            for (size_t step = 0; 
                                step < SUBPIXEL_FITTING_MAX_STEPS && !done &&
                                    0 <     j + dy &&     j + dy + 1 < DoGs[1].rows && 
                                    0 <     i + dx &&     i + dx + 1 < DoGs[1].cols; 
                                ++step
                            ) {
                                const cv::Vec3f gradient {
                                    (DoGs[1].at<float>(j + dy    , i + dx + 1) - DoGs[1].at<float>(j + dy    , i + dx - 1)) * 0.5f,
                                    (DoGs[1].at<float>(j + dy + 1, i + dx    ) - DoGs[1].at<float>(j + dy - 1, i + dx    )) * 0.5f,
                                    (DoGs[2].at<float>(j + dy    , i + dx    ) - DoGs[0].at<float>(j + dy    , i + dx    )) * 0.5f
                                };

                                const float dxdx =   DoGs[1].at<float>(j + dy    , i + dx + 1) + DoGs[1].at<float>(j + dy    , i + dx - 1) - center * 2; // one dimensional second derivative -- can compute easier
                                const float dydy =   DoGs[1].at<float>(j + dy + 1, i + dx    ) + DoGs[1].at<float>(j + dy - 1, i + dx    ) - center * 2;
                                const float dzdz =   DoGs[2].at<float>(j + dy    , i + dx    ) + DoGs[0].at<float>(j + dy    , i + dx    ) - center * 2;
                                const float dxdy = ((DoGs[1].at<float>(j + dy + 1, i + dx + 1) - DoGs[1].at<float>(j + dy + 1, i + dx - 1)) - 
                                                    (DoGs[1].at<float>(j + dy - 1, i + dx + 1) - DoGs[1].at<float>(j + dy - 1, i + dx - 1))) * 0.25f; // 2dim partial derivative <=> partial derivative of 1dim partial derivatives
                                const float dxdz = ((DoGs[2].at<float>(j + dy    , i + dx + 1) - DoGs[2].at<float>(j + dy    , i + dx - 1)) - 
                                                    (DoGs[0].at<float>(j + dy    , i + dx + 1) - DoGs[0].at<float>(j + dy    , i + dx - 1))) * 0.25f;
                                const float dydz = ((DoGs[2].at<float>(j + dy + 1, i + dx    ) - DoGs[2].at<float>(j + dy - 1, i + dx    )) - 
                                                    (DoGs[0].at<float>(j + dy + 1, i + dx    ) - DoGs[0].at<float>(j + dy - 1, i + dx    ))) * 0.25f;
                                
                                const cv::Matx33f hessian {
                                    dxdx, dxdy, dxdz,
                                    dxdy, dydy, dydz,
                                    dxdz, dydz, dzdz
                                };

                                delta = hessian.solve(-gradient, cv::DECOMP_LU);
                                
                                const auto try_adjust = [](int &dcell, float &dcoord) {
                                    if (std::abs(dcoord) < 0.5) {
                                        return false;
                                    }

                                    dcell += std::round(dcoord);
                                    return true;
                                };

                                const float fixed_value = DoGs[1].at<float>(j + dy, i + dx) + gradient.dot(cv::Vec3f(delta[0], delta[1], 0)) * 0.5f;
                                done |= !(try_adjust(dx, delta[0]) || try_adjust(dy, delta[1])) && 
                                        0 < j + dy && j + dy + 1 < DoGs[1].rows && 
                                        0 < i + dx && i + dx + 1 < DoGs[1].cols;

                                delta[2] = fixed_value;
                            }

                            if (!done) 
                                continue;

                            delta[0] += dx;
                            delta[1] += dy;
                        }

                        const float dx    = delta[0];
                        const float dy    = delta[1];
                        const float value = delta[2];
#else
                        const float dx = 0.0f;
                        const float dy = 0.0f;
                        const float value = center;

#endif
                        cv::KeyPoint kp;

                        kp.pt = cv::Point2f((i + 0.5 + dx) * octave_downscale, (j + 0.5 + dy) * octave_downscale);

                        kp.response = fabs(value);

                        if (kp.response < contrast_threshold / OCTAVE_NLAYERS)
                            continue;

                        #ifdef CURVATURE_THRESHOLD
                        { // отфильтруем ещё по "похожести" на границу
                            const double new_center = DoGs[1].at<float>(j + dy, i + dx);
                            const double dxdx =   DoGs[1].at<float>(j + dy    , i + dx + 1) + DoGs[1].at<float>(j + dy    , i + dx - 1) - 2 * new_center;
                            const double dydy =   DoGs[1].at<float>(j + dy + 1, i + dx    ) + DoGs[1].at<float>(j + dy    , i + dx - 1) - 2 * new_center;
                            const double dxdy = ((DoGs[1].at<float>(j + dy + 1, i + dx + 1) - DoGs[1].at<float>(j + dy + 1, i + dx - 1)) - 
                                                 (DoGs[1].at<float>(j + dy - 1, i + dx + 1) - DoGs[1].at<float>(j + dy - 1, i + dx - 1))) * 0.25f;

                            const double trace       = dxdx + dydy;
                            const double determinant = dxdx * dydy - dxdy * dxdy;

                            if (
                                determinant <= 0 || 
                                trace * trace / determinant >= (CURVATURE_THRESHOLD + 1.0) * (CURVATURE_THRESHOLD + 1.0 ) / CURVATURE_THRESHOLD
                            ) {
                                continue;
                            }
                        }
                        #endif

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS
                        const double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
                        kp.size = 2.0 * sigmaCur * 5.0;

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[layer_idx(octave, layer)];
                        std::vector<float> votes;
                        float biggestVote;
                        const int oriRadius = (int) (ORIENTATION_WINDOW_R * (1.0 + k * (layer - 1)));
                        if (!buildLocalOrientationHists(img, i, j, oriRadius, votes, biggestVote))
                            continue;

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
                            if (value > prevValue && value > nextValue && votes[bin] > biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                kp.angle = fix_angle((bin + parabolaFitting(prevValue, value, nextValue) + 0.5) * (360.0 / ORIENTATION_NHISTS));
                                rassert(kp.angle >= 0.0 && kp.angle <= 360.0, 123512412412);
                                
                                std::vector<float> descriptor;
                                double descrSampleRadius = (DESCRIPTOR_SAMPLE_WINDOW_R * (1.0 + k * (layer - 1)));
                                if (!buildDescriptor(img, kp.pt.x, kp.pt.y, descrSampleRadius, kp.angle, descriptor))
                                    continue;
                                
                                thread_points.push_back(kp);
                                thread_descriptors.push_back(descriptor);
                            }
                        }
                    }
                }
            }
        }

        // в критической секции объединяем все массивы детектированных точек
        #pragma omp critical
        {
            keyPoints.insert(keyPoints.end(), thread_points.begin(), thread_points.end());
            pointsDesc.insert(pointsDesc.end(), thread_descriptors.begin(), thread_descriptors.end());
        }
    }

    rassert(pointsDesc.size() == keyPoints.size(), 12356351235124);
    desc = cv::Mat(pointsDesc.size(), DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, CV_32FC1);
    for (size_t j = 0; j < pointsDesc.size(); ++j) {
        rassert(pointsDesc[j].size() == DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 1253351412421);
        for (size_t i = 0; i < pointsDesc[i].size(); ++i) {
            desc.at<float>(j, i) = pointsDesc[j][i];
        }
    }
}

bool phg::SIFT::buildLocalOrientationHists(const cv::Mat &img, size_t i, size_t j, size_t radius,
                                           std::vector<float> &votes, float &biggestVote) {
    // 5 Orientation assignment
    votes.resize(ORIENTATION_NHISTS, 0.0f);
    biggestVote = 0.0;

    if (i-1 < radius - 1 || i+1 + radius - 1 >= img.cols || j-1 < radius - 1 || j+1 + radius - 1 >= img.rows)
        return false;

    cv::Mat sum(ORIENTATION_NHISTS, 1, CV_32FC1, 0.0f);

    for (size_t y = j - radius + 1; y < j + radius; ++y) {
        for (size_t x = i - radius + 1; x < i + radius; ++x) {
            // m(x, y)=(L(x + 1, y) − L(x − 1, y))^2 + (L(x, y + 1) − L(x, y − 1))^2
            const cv::Vec2f gradient {
                img.at<float>(j, i + 1) - img.at<float>(j, i - 1),
                img.at<float>(j + 1, i) - img.at<float>(j - 1, i)
            };

            const double magnitude = gradient.dot(gradient);
            const double orientation = atan2deg(gradient[1], gradient[0]);

            rassert(orientation >= 0.0 && orientation < 360.0, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");
            size_t bin = orientation / ORIENTATION_NHISTS;
            rassert(bin < ORIENTATION_NHISTS, 361236315613);
            sum.at<float>(bin) += magnitude;
        }
    }

    #if ORIENTATION_HISTOGRAM_SMOOTH_DISTANCE
    // может быть, сгладить получившиеся гистограммы улучшит результат? E
    cv::GaussianBlur(sum, sum, cv::Size(ORIENTATION_HISTOGRAM_SMOOTH_DISTANCE, 1), 0);
    #endif

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum.at<float>(bin);
        biggestVote = std::max(biggestVote, votes[bin]);
    }

    return true;
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
    cv::Mat relativeShiftRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, 1.0);

    const double smpW = 2.0 * descrSampleRadius - 1.0;

    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);
    for (int hstj = 0; hstj < DESCRIPTOR_SIZE; ++hstj) { // перебираем строку в решетке гистограмм
        for (int hsti = 0; hsti < DESCRIPTOR_SIZE; ++hsti) { // перебираем колонку в решетке гистограмм

            cv::Mat sum(DESCRIPTOR_NBINS, 1, CV_32FC1, 0.0f);

            for (int smpj = 0; smpj < DESCRIPTOR_SAMPLES_N; ++smpj) {     // перебираем строчку замера для текущей гистограммы
                for (int smpi = 0; smpi < DESCRIPTOR_SAMPLES_N; ++smpi) { // перебираем столбик очередного замера для текущей гистограммы
                    for (int smpy = 0; smpy < smpW; ++smpy) {             // перебираем ряд пикселей текущего замера
                        for (int smpx = 0; smpx < smpW; ++smpx) {         // перебираем столбик пикселей текущего замера
                            cv::Point2f shift(
                                ((-DESCRIPTOR_SIZE/2.0 + hsti) * DESCRIPTOR_SAMPLES_N + smpi) * smpW, 
                                ((-DESCRIPTOR_SIZE/2.0 + hstj) * DESCRIPTOR_SAMPLES_N + smpj) * smpW
                            );
                            std::vector<cv::Point2f> shiftInVector(1, shift);
                            cv::transform(shiftInVector, shiftInVector, relativeShiftRotation); // преобразуем относительный сдвиг с учетом ориентации ключевой точки
                            shift = shiftInVector[0];
 
                            int x = (int) (px + shift.x);
                            int y = (int) (py + shift.y);
 
                            if (y - 1 < 0 || y + 1 >= img.rows || x - 1 < 0 || x + 1 >= img.cols)
                                return false;

                            const cv::Vec2f gradient {
                                img.at<float>(y    , x + 1) - img.at<float>(y    , x - 1),
                                img.at<float>(y + 1, x    ) - img.at<float>(y - 1, x    )
                            }; // делить на dx, dy не нужно, так как atan2 пофиг на домножение на одинаковую константу

                            const double magnitude = gradient.dot(gradient);
                            const double orientation = atan2deg(gradient[1], gradient[0]); // уже повернули всю область -> ничего делать не надо
                            rassert(0.0 <= orientation && orientation < 360.0, 3515215125412);

                            static_assert(360 % DESCRIPTOR_NBINS == 0, "Inappropriate bins number!");
                            const size_t bin = orientation / DESCRIPTOR_NBINS;

                            const auto add_weighed = [&](size_t bin) {
                                #if DESCRIPTOR_INTERPOLATE_HISTS
                                bin %= DESCRIPTOR_NBINS;
                                const double bin_center = (bin + 0.5)  * DESCRIPTOR_NBINS;
                                const double weight = std::max(0.0, std::abs(bin_center - orientation));
                                #else 
                                const double weight = 1;
                                #endif

                                rassert(bin < DESCRIPTOR_NBINS, 7389217389127);
                                sum.at<float>(bin) += magnitude * weight;
                            };

                            add_weighed(bin);
                            #if DESCRIPTOR_INTERPOLATE_HISTS
                            add_weighed(bin + 1);
                            add_weighed(bin + DESCRIPTOR_NBINS - 1); // bin - 1
                            #endif
                        }
                    }
                }
            }

            sum /= cv::norm(sum);

            float *votes = &(descriptor[(hstj * DESCRIPTOR_SIZE + hsti) * DESCRIPTOR_NBINS]); // нашли где будут лежать корзины нашей гистограммы
            for (int bin = 0; bin < DESCRIPTOR_NBINS; ++bin) {
                votes[bin] = sum.at<float>(bin);
            }
        }
    }

    return true;
}
