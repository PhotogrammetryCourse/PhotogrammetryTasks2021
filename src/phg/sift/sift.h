#pragma once

#include <vector>

#include <opencv2/core.hpp>


namespace phg {
    class SIFT {

    public:
        // Можете добавить дополнительных параметров со значениями по умолчанию в конструктор если хотите
        SIFT(double contrast_threshold = 2.0) : contrast_threshold(contrast_threshold) {}

        // Сигнатуру этого метода менять нельзя
        void detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc);

    protected: // Можете менять внутренние детали реализации включая разбиение на эти методы (это просто набросок):

        static void buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid);

        void findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                          std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) const;

        static void getPointOrientationAndDescriptor(const cv::Mat &gaussPic, const cv::Point2i &pix, int layer,
                                              const cv::Point2f &point, float sigma, float contrast,
                                              std::vector<cv::KeyPoint> &points, std::vector<cv::Mat> &descriptors);

        static cv::Mat buildDescriptor(const cv::Mat &gaussPic, const cv::Point2i &pix, const cv::KeyPoint &keyPoint,
                                       int layer);

        double contrast_threshold;
    };

}
