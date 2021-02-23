#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams();
    search_params = flannKsTreeSearchParams();
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    size_t n_desc = query_desc.rows;
    cv::Mat indices(n_desc, k, CV_32SC1);
    cv::Mat distances2(n_desc, k, CV_32FC1);

    flann_index->knnSearch(query_desc, indices, distances2, k, *search_params);

    matches.resize(n_desc, std::vector<cv::DMatch>(k));
    for (int i = 0; i < n_desc; ++i) {
        for (int j = 0; j < k; ++j) {
            matches[i][j].distance = distances2.at<float>(i, j);
            matches[i][j].queryIdx = i;
            matches[i][j].trainIdx = indices.at<int>(i, j);
        }
    }
}
