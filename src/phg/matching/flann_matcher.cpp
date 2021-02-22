#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(5);
    search_params = flannKsTreeSearchParams(10);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    cv::Mat indices(query_desc.rows, k, CV_32S);
    cv::Mat dists(query_desc.rows, k, CV_32F);

    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);

    matches.resize(query_desc.rows);
    for (int i = 0; i < query_desc.rows; ++i) {
        matches[i].resize(k);
        for (int j = 0; j < k; ++j) {
            matches[i][j].distance = dists.at<float>(i, j);
            matches[i][j].queryIdx = i;
            matches[i][j].trainIdx = indices.at<int>(i, j);
        }
    }
}
