#include "panorama_stitcher.h"
#include "homography.h"

#include <libutils/bbox2.h>
#include <iostream>
#include <stack>

/*
 * imgs - список картинок
 * parent - список индексов, каждый индекс указывает, к какой картинке должна быть приклеена текущая картинка
 *          этот список образует дерево, корень дерева (картинка, которая ни к кому не приклеивается, приклеиваются только к ней), в данном массиве имеет значение -1
 * homography_builder - функтор, возвращающий гомографию по паре картинок
 * */
cv::Mat phg::stitchPanorama(const std::vector<cv::Mat> &imgs,
                            const std::vector<int> &parent,
                            std::function<cv::Mat(const cv::Mat &, const cv::Mat &)> &homography_builder)
{
    const int n_images = imgs.size();

    // склеивание панорамы происходит через приклеивание всех картинок к корню, некоторые приклеиваются не напрямую, а через цепочку других картинок

    // вектор гомографий, для каждой картинки описывает преобразование до корня
    std::vector<cv::Mat> Hs(n_images);
    {
        std::vector<int> init(n_images, 0);
        for (int i = 0 ; i < n_images; ++i) {
            if (init[i] == 1) continue;
            if (parent[i] == -1) {
                Hs[i] = cv::Mat::eye(3, 3, 6);
                init[i] = 1;
                continue;
            }
            // имитируем рекурсию стеком
            std::stack<int> im_indexes;
            int current = i;
            im_indexes.push(current);
            while (init[current] != 1 && parent[current] != -1) {
                current = parent[current];
                im_indexes.push(current);
            }
            im_indexes.pop();
            while (!im_indexes.empty()) {
                int child = im_indexes.top();
                im_indexes.pop();

                if (parent[current] == -1) {
                    Hs[child] = homography_builder(imgs[child], imgs[current]);
                } else {
                    Hs[child] = Hs[current] * homography_builder(imgs[child], imgs[current]);
                }

                init[child] = 1;
                current = child;
            }
        }
    }

    bbox2<double, cv::Point2d> bbox;
    for (int i = 0; i < n_images; ++i) {
        double w = imgs[i].cols;
        double h = imgs[i].rows;
        bbox.grow(phg::transformPoint(cv::Point2d(0.0, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, 0.0), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(w, h), Hs[i]));
        bbox.grow(phg::transformPoint(cv::Point2d(0, h), Hs[i]));
    }

    std::cout << "bbox: " << bbox.max() << ", " << bbox.min() << std::endl;

    int result_width = bbox.width() + 1;
    int result_height = bbox.height() + 1;

    cv::Mat result = cv::Mat::zeros(result_height, result_width, CV_8UC3);

    // из-за растяжения пикселей при использовании прямой матрицы гомографии после отображения между пикселями остается пустое пространство
    // лучше использовать обратную и для каждого пикселя на итоговвой картинке проверять, с какой картинки он может получить цвет
    // тогда в некоторых пикселях цвет будет дублироваться, но изображение будет непрерывным
//        for (int i = 0; i < n_images; ++i) {
//            for (int y = 0; y < imgs[i].rows; ++y) {
//                for (int x = 0; x < imgs[i].cols; ++x) {
//                    cv::Vec3b color = imgs[i].at<cv::Vec3b>(y, x);
//
//                    cv::Point2d pt_dst = applyH(cv::Point2d(x, y), Hs[i]) - bbox.min();
//                    int y_dst = std::max(0, std::min((int) std::round(pt_dst.y), result_height - 1));
//                    int x_dst = std::max(0, std::min((int) std::round(pt_dst.x), result_width - 1));
//
//                    result.at<cv::Vec3b>(y_dst, x_dst) = color;
//                }
//            }
//        }

    std::vector<cv::Mat> Hs_inv;
    std::transform(Hs.begin(), Hs.end(), std::back_inserter(Hs_inv), [&](const cv::Mat &H){ return H.inv(); });

#pragma omp parallel for
    for (int y = 0; y < result_height; ++y) {
        for (int x = 0; x < result_width; ++x) {

            cv::Point2d pt_dst(x, y);

            // test all images, pick first
            for (int i = 0; i < n_images; ++i) {

                cv::Point2d pt_src = phg::transformPoint(pt_dst + bbox.min(), Hs_inv[i]);

                int x_src = std::round(pt_src.x);
                int y_src = std::round(pt_src.y);

                if (x_src >= 0 && x_src < imgs[i].cols && y_src >= 0 && y_src < imgs[i].rows) {
                    result.at<cv::Vec3b>(y, x) = imgs[i].at<cv::Vec3b>(y_src, x_src);
                    break;
                }
            }

        }
    }

    return result;
}
