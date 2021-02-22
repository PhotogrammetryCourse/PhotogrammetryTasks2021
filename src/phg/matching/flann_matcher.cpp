#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(5);
    search_params = flannKsTreeSearchParams(29);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    matches.clear();

    cv::Mat indices, dists;
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);
    for (int q_idx = 0; q_idx < query_desc.rows; ++q_idx) {
        std::vector<cv::DMatch> neighbors;
        for (int nbr = 0; nbr < k; ++nbr) {
            neighbors.emplace_back(q_idx, indices.at<int32_t>(q_idx, nbr), dists.at<float>(q_idx, nbr));
        }
        matches.push_back(neighbors);
    }
}
