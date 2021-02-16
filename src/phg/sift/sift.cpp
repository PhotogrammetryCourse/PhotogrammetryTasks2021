#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <libutils/rasserts.h>

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
   
#define SUBPIXEL_FITTING_ENABLE      1   // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить

#define EXTREMUM_THRESHOLD           250
#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно


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
    std::vector<double> sigmas(NOCTAVES * OCTAVE_GAUSSIAN_IMAGES);


    // строим пирамиду гауссовых размытий картинки
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        if (octave == 0) {
            int layer = 0;
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgOrg.clone();
            sigmas[octave] = INITIAL_IMG_SIGMA;
//            std::cerr << sigmas[octave] << " " << octave << "\n";
            for (int i = 1; i < OCTAVE_GAUSSIAN_IMAGES; ++i) {
                sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + i] = sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + layer] * std::pow(2.0, std::floor(i) / 3.0);
//                std::cerr << sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + i] << " " << octave * OCTAVE_GAUSSIAN_IMAGES + i << "\n";
            }
        } else {
            int layer = 0;
            size_t prevOctave = octave - 1;
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия (сигмы должны совпадать)
            int ind = prevOctave * OCTAVE_GAUSSIAN_IMAGES + OCTAVE_NLAYERS;
            cv::Mat img = gaussianPyramid[ind].clone();
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет не идеально 2 пикселя в один схлопываться - а слегка смещаться
            cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2), 0.5, 0.5, cv::INTER_NEAREST);
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = img;
            sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = INITIAL_IMG_SIGMA * 2;
//            std::cerr << sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + layer] << " " << octave * OCTAVE_GAUSSIAN_IMAGES + layer << "\n";
            for (int i = 1; i < OCTAVE_GAUSSIAN_IMAGES; ++i) {
                sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + i] = sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + layer] * std::pow(2.0, std::floor(i) / 3.0);
//                std::cerr << sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + i] << " " << octave * OCTAVE_GAUSSIAN_IMAGES + i << "\n";
            }
        }

        #pragma omp parallel for
// TODO: если выполните TODO про "размытие из изначального слоя октавы" ниже - раскоментируйте это распараллеливание, ведь теперь слои считаются независимо (из самого первого), проверьте что результат на картинках не изменился
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
//            size_t prevLayer = layer - 1;
//
//            // если есть два последовательных гауссовых размытия с sigma1 и sigma2, то результат будет с sigma12=sqrt(sigma1^2 + sigma2^2) => sigma2=sqrt(sigma12^2-sigma1^2)
//            double sigmaPrev = INITIAL_IMG_SIGMA * pow(k, prevLayer); // sigma1  - сигма до которой дошла картинка на предыдущем слое
//            double sigmaCur  = INITIAL_IMG_SIGMA * pow(k, layer);     // sigma12 - сигма до которой мы хотим дойти на текущем слое
//            double sigma = sqrt(sigmaCur*sigmaCur - sigmaPrev*sigmaPrev);                // sigma2  - сигма которую надо добавить чтобы довести sigma1 до sigma12
//
//            // TODO: переделайте это добавочное размытие с варианта "размываем предыдущий слой" на вариант "размываем самый первый слой октавы до степени размытия сигмы нашего текущего слоя"
//            // проверьте - картинки отладочного вывода выглядят один-в-один до/после? (посмотрите на них туда-сюда быстро мигая)
//
//            cv::Mat imgLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer].clone();
//            cv::Size automaticKernelSize = cv::Size(0, 0);
//
//            cv::GaussianBlur(imgLayer, imgLayer, automaticKernelSize, sigma, sigma);
//
//            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgLayer;
            cv::Mat imgLayer = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES].clone();
            cv::Size automaticKernelSize = cv::Size(0, 0);
            double sigma = sigmas[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
            cv::GaussianBlur(imgLayer, imgLayer, automaticKernelSize, sigma, sigma);
            gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer] = imgLayer;
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/new_o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer]);
            // TODO: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // спойлер: подуймайте с чем должна визуально совпадать картинка из октавы? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера? 
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);

    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG), т.к. вычитать надо из слоя слой в рамках одной и той же октавы - то есть приятный параллелизм на уровне октав
    #pragma omp parallel for
    for (ptrdiff_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            int prevLayer = layer - 1;
            cv::Mat imgPrevGaussian = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer];
            cv::Mat imgCurGaussian  = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];

            cv::Mat imgCurDoG = imgCurGaussian.clone();
            // обратите внимание что т.к. пиксели картинки из одного ряда лежат в памяти подряд, поэтому если вложенный цикл бежит подряд по одному и тому же ряду
            // то код работает быстрее т.к. он будет более cache-friendly, можете сравнить оценить ускорение добавив замер времени построения пирамиды: timer t; double time_s = t.elapsed();
            for (size_t j = 0; j < imgCurDoG.rows; ++j) {
                for (size_t i = 0; i < imgCurDoG.cols; ++i) {
//                    if (imgCurGaussian.at<float>(j, i) - imgPrevGaussian.at<float>(j, i))
//                    std::cerr << imgCurGaussian.at<float>(j, i) << " " << imgPrevGaussian.at<float>(j, i) << "\n";
                    imgCurDoG.at<float>(j, i) = 100 * (imgCurGaussian.at<float>(j, i) - imgPrevGaussian.at<float>(j, i));
                }
            }
            int dogLayer = layer - 1;
            DoGPyramid[octave * OCTAVE_DOG_IMAGES + dogLayer] = imgCurDoG;
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer - 1);
//            if (DEBUG_ENABLE) {
//                cv::imwrite(DEBUG_PATH + "pyramidDoG/prev_o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", imgPrevGaussian);
//                cv::imwrite(DEBUG_PATH + "pyramidDoG/curr_o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", imgCurGaussian);
//                cv::imwrite(DEBUG_PATH + "pyramidDoG/new_o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", imgCurDoG);
//            }

        }
    }

    // нам нужны padding-картинки по краям октавы чтобы извлекать экстремумы, но в статье предлагается не s+2 а s+3: [lowe04] We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema detection covers a complete octave
    // TODO: почему OCTAVE_GAUSSIAN_IMAGES=(OCTAVE_NLAYERS + 3) а не например (OCTAVE_NLAYERS + 2)?

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);

            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramidDoG/err_o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer]);
            // TODO: какие ожидания от картинок можно придумать? т.е. как дополнительно проверить что работает разумно?
            // спойлер: подуймайте с чем должна визуально совпадать картинка из октавы DoG? может быть с какой-то из картинок с предыдущей октавы? с какой? как их визуально сверить ведь они разного размера? 
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
                cv::Mat prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1];
                cv::Mat cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                cv::Mat next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1];
                cv::Mat DoGs[3] = {prev, cur, next};

                // теперь каждый поток обработает свой кусок картинки 
                #pragma omp for
                for (ptrdiff_t j = 1; j < cur.rows - 1; ++j) {
                    for (ptrdiff_t i = 1; i + 1 < cur.cols; ++i) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = DoGs[1].at<float>(j, i);
                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                        for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                        for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                            if (dx != 0 && dy != 0 && dz != 0) {
                                if (DoGs[1 + dz].at<float>(j + dy, i + dx) > center) {
                                    is_max = false;
                                } else if (DoGs[1 + dz].at<float>(j + dy, i + dx) < center) {
                                    is_min = false;
                                }
                            }

                            // TODO проверить является ли наш центр все еще экстремум по сравнению с соседом DoGs[1+dz].at<float>(j+dy, i+dx) ? (не забудьте учесть что один из соседов - это мы сами)
                        }
                        }
                        }
                        bool is_extremum = (is_min || is_max);
//                        if (is_extremum)
//                            std::cerr << i << " " << j << "\n";

                        if (!is_extremum)
                            continue;
                        // очередной элемент cascade filtering, если не экстремум - сразу заканчиваем обработку этого пикселя
//                        std::cerr << center << "\n";
                        if (fabs(center) < EXTREMUM_THRESHOLD)
                            continue;
                            // 4 Accurate keypoint localization
                        cv::KeyPoint kp;
                        float dx = 0.0f;
                        float dy = 0.0f;
                        float dvalue = 0.0f;
                        float x_i, x_j, x_o = 0.0f;
                        // TODO сделать субпиксельное уточнение (хотя бы через параболу-фиттинг независимо по оси X и оси Y, но лучше через честный ряд Тейлора, матрицу Гессе и итеративное смещение если экстремум оказался в соседнем пикселе)
#if SUBPIXEL_FITTING_ENABLE // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить
                        {
                            int curr_j = j;
                            int curr_i = i;
                            int curr_o = layer;
                            float diff_i, diff_j, diff_o = 0.0f;
                            int max_iter = 3;
                            int cnt = 0;
                            while (fabs(curr_j - j) <= 2 && fabs(curr_i - i) <= 2 && fabs(curr_o - layer) <= 2 && cnt < max_iter){

                                cnt ++;
                                float v2 = (float)cur.at<float>(curr_j, curr_i)*2;
                                float dxx = (cur.at<float>(curr_j, curr_i+1) + cur.at<float>(curr_j, curr_i-1) - v2);
                                float dyy = (cur.at<float>(curr_j+1, curr_i) + cur.at<float>(curr_j-1, curr_j) - v2);
                                float dss = (next.at<float>(curr_j, curr_i) + prev.at<float>(curr_j, curr_i) - v2);
                                float dxy = (cur.at<float>(curr_j+1, curr_i+1) - cur.at<float>(curr_j+1, curr_i-1) -
                                             cur.at<float>(curr_j-1, curr_i+1) + cur.at<float>(curr_j-1, curr_i-1));
                                float dxs = (next.at<float>(curr_j, curr_i+1) - next.at<float>(curr_j, curr_i-1) -
                                             prev.at<float>(curr_j, curr_i+1) + prev.at<float>(curr_j, curr_i-1));
                                float dys = (next.at<float>(curr_j+1, curr_i) - next.at<float>(curr_j-1, curr_i) -
                                             prev.at<float>(curr_j+1, curr_i) + prev.at<float>(curr_j-1, curr_i));

                                cv::Vec3f dD ((cur.at<float>(curr_j, curr_i+1) - cur.at<float>(curr_j, curr_i-1)),
                                              (cur.at<float>(curr_j+1, curr_i) - cur.at<float>(curr_j-1, curr_i)),
                                              (next.at<float>(curr_j, curr_i) - prev.at<float>(curr_j, curr_i)));

                                cv::Matx33f H(dxx, dxy, dxs,
                                              dxy, dyy, dys,
                                              dxs, dys, dss);
                                cv::Mat x_hat = (cv::Mat) H.solve(dD);
//                                std::cerr << x_hat.size << " " << x_hat.rows << " " << x_hat.cols << "\n";
                                x_o = -x_hat.at<float>(2, 0);
                                x_j = -x_hat.at<float>(1, 0);
                                x_i = -x_hat.at<float>(0, 0);
//                                std::cerr << x_o << " " << x_j << " " << x_i << "\n";
                                if( fabs(x_i) < 0.5f && fabs(x_j) < 0.5f && fabs(x_o) < 0.5f )
                                    break;

                                if( fabs(x_i) > 2.0f || fabs(x_j) > 2.0f || fabs(x_o) > 2.0f )
                                    break;

                                if( fabs(x_i) > (float) (INT_MAX / 3) ||
                                fabs(x_j) > (float) (INT_MAX / 3) ||
                                fabs(x_o) > (float) (INT_MAX / 3) )
                                    break;

                                diff_i = cvRound(x_i);
                                diff_j = cvRound(x_j);
                                diff_o = cvRound(x_o);

                                curr_j += diff_j;
                                curr_i += diff_i;
                                curr_o += diff_o;

//                                std::cerr << curr_i << " " << curr_j << " " << curr_o << "\n";

                                dvalue = dD.dot(cv::Matx31f(x_i, x_j, x_o));

                                if (curr_o < 1 || curr_o >= OCTAVE_DOG_IMAGES - 1)
                                    break;
                                if (curr_i < 1 || curr_i >= cur.cols - 1)
                                    break;
                                if (curr_j < 1 || curr_j >= cur.rows - 1)
                                    break;
                                prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + curr_o - 1];
                                cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + curr_o];
                                next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + curr_o + 1];
                                center = cur.at<float>(curr_j, curr_i);
                            }
                            // TODO
                        }
#endif
                        // TODO сделать фильтрацию слабых точек по слабому контрасту
                        float contrast = fabs(center + dvalue);
//                        std::cerr << contrast << "\n";
                        if (contrast < contrast_threshold / OCTAVE_NLAYERS) // TODO почему порог контрастности должен уменьшаться при увеличении числа слоев в октаве?
                            continue;

                        kp.pt = cv::Point2f((i + 0.5 + dx) * octave_downscale, (j + 0.5 + dy) * octave_downscale);

                        kp.response = fabs(contrast);

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS
                        double sigmaCur = INITIAL_IMG_SIGMA * pow(2, octave) * pow(k, layer);
                        kp.size = 2.0 * sigmaCur * 5.0;

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
                        std::vector<float> votes;
                        float biggestVote;
                        int oriRadius = (int) (ORIENTATION_WINDOW_R * (1.0 + k * (layer - 1)));
                        if (!buildLocalOrientationHists(img, i, j, oriRadius, votes, biggestVote))
                            continue;

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
//                            std::cerr << value << " " << value << " " << nextValue << "\n";
                            if (value > prevValue && value > nextValue && votes[bin] > biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                float dangle = parabolaFitting(prevValue, value, nextValue);
                                // TODO добавьте уточнение угла наклона - может помочь определенная выше функция parabolaFitting(float x0, float x1, float x2)
                                kp.angle = (bin + dangle + 0.5) * (360.0 / ORIENTATION_NHISTS);
//                                std::cerr << dangle << "\n";

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

    float sum[ORIENTATION_NHISTS] = {0.0f};

    for (size_t y = j - radius + 1; y < j + radius; ++y) {
        for (size_t x = i - radius + 1; x < i + radius; ++x) {
            // m(x, y)=(L(x + 1, y) − L(x − 1, y))^2 + (L(x, y + 1) − L(x, y − 1))^2
            float dx = (float)(img.at<float>(y, x+1) - img.at<float>(y, x-1));
            float dy = (float)(img.at<float>(y-1, x) - img.at<float>(y+1, x));
//            std::cerr << y + j << " " << y + j - 1 << " " << y + j + 1 << " " << i + x + 1 << " " << i + x - 1 << " " << x + i << "\n";
            double magnitude = std::sqrt(dx * dx + dy + dy);

            // orientation == theta
            // atan( (L(x, y + 1) − L(x, y − 1)) / (L(x + 1, y) − L(x − 1, y)) )
            double orientation = atan2(dy, dx);
            orientation = orientation * 180.0 / M_PI;
            orientation = (orientation + 90.0);
            if (orientation <  0.0)   orientation += 360.0;
            if (orientation >= 360.0) orientation -= 360.0;
//            rassert(orientation >= 0.0 && orientation < 360.0, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");
            size_t bin = (int) (orientation / (360.0 / ORIENTATION_NHISTS));
//            rassert(bin < ORIENTATION_NHISTS, 361236315613);
//            std::cerr << " " << orientation << "\n";
            sum[bin] += magnitude;
//            std::cerr << sum[bin] << " " << bin << "\n";
            // TODO может быть сгладить получившиеся гистограммы улучшит результат? 
        }
    }

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum[bin];
        biggestVote = std::max(biggestVote, sum[bin]);
    }
//    std::cerr << biggestVote << "\n";

    return true;
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
    cv::Mat relativeShiftRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, 1.0);

    const double smpW = 2.0 * descrSampleRadius - 1.0;
//    std::cerr << smpW << "\n";

    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);
    for (int hstj = 0; hstj < DESCRIPTOR_SIZE; ++hstj) { // перебираем строку в решетке гистограмм
        for (int hsti = 0; hsti < DESCRIPTOR_SIZE; ++hsti) { // перебираем колонку в решетке гистограмм

            float sum[DESCRIPTOR_NBINS] = {0.0f};

            for (int smpj = 0; smpj < DESCRIPTOR_SAMPLES_N; ++smpj) { // перебираем строчку замера для текущей гистограммы
                for (int smpi = 0; smpi < DESCRIPTOR_SAMPLES_N; ++smpi) { // перебираем столбик очередного замера для текущей гистограммы
                    for (int smpy = 0; smpy < smpW; ++smpy) { // перебираем ряд пикселей текущего замера
                        for (int smpx = 0; smpx < smpW; ++smpx) { // перебираем столбик пикселей текущего замера

                            cv::Point2f shift(((-DESCRIPTOR_SIZE/2.0 + hsti) * DESCRIPTOR_SAMPLES_N + smpi) * smpW,
                                              ((-DESCRIPTOR_SIZE/2.0 + hstj) * DESCRIPTOR_SAMPLES_N + smpj) * smpW);
                            std::vector<cv::Point2f> shiftInVector(1, shift);
                            cv::transform(shiftInVector, shiftInVector, relativeShiftRotation); // преобразуем относительный сдвиг с учетом ориентации ключевой точки
                            shift = shiftInVector[0];
//
                            int x = (int) (px + shift.x);
                            int y = (int) (py + shift.y);
//
                            if (y - 1 < 0 || y + 1 >= img.rows || x - 1 < 0 || x + 1 >= img.cols)
                                return false;
                            float dx = (float)(img.at<float>(y, x+1) - img.at<float>(y, x-1));
                            float dy = (float)(img.at<float>(y-1, x) - img.at<float>(y+1, x));
                            double magnitude = std::sqrt(dx * dx + dy * dy);
//
                            double orientation = atan2(dy, dx);
                            orientation = orientation * 180.0 / M_PI;
                            orientation = (orientation + 90.0);

                            orientation -= angle;
                            if (orientation <  0.0)   orientation += 360.0;
                            if (orientation >= 360.0) orientation -= 360.0;

//                              // TODO за счет чего этот вклад будет сравниваться с этим же вкладом даже если эта картинка будет повернута? что нужно сделать с ориентацией каждого градиента из окрестности этой ключевой точки?
//
//                            rassert(orientation >= 0.0 && orientation < 360.0, 3515215125412);
                            static_assert(360 % DESCRIPTOR_NBINS == 0, "Inappropriate bins number!");
                            size_t bin = (int) orientation / (360.0 / DESCRIPTOR_NBINS);
//                            rassert(bin < DESCRIPTOR_NBINS, 361236315613);
                            sum[bin] += magnitude;
//                            std::cerr << sum[bin] << " " << bin << "\n";
//                            // TODO хорошая идея добавить трилинейную интерполяцию как предложено в статье, или хотя бы сэмулировать ее - сгладить получившиеся гистограммы
                        }
                    }
                }
            }

            // TODO нормализовать наш вектор дескриптор (подсказка: посчитать его длину и поделить каждую компоненту на эту длину)

            float *votes = &(descriptor[(hstj * DESCRIPTOR_SIZE + hsti) * DESCRIPTOR_NBINS]); // нашли где будут лежать корзины нашей гистограммы
            for (int bin = 0; bin < DESCRIPTOR_NBINS; ++bin) {
                votes[bin] = sum[bin];
            }
//            for (int i = 0; i < DESCRIPTOR_NBINS; ++i) {
//                std::cerr << votes[i] << " ";
//            }
//            std::cerr << "\n";
        }
    }
//    std::cerr << descriptor.size() << "\n";
    double s = 0;
    for (int i = 0; i < descriptor.size(); ++i) {
        s += descriptor[i] * descriptor[i];
    }
    double sq = 0;
    sq = std::sqrt(s);
//    if (sq < 15000)
//        return false;
    for (int i = 0; i < descriptor.size(); ++i) {
        descriptor[i] /= sq;
    }
    return true;
}
