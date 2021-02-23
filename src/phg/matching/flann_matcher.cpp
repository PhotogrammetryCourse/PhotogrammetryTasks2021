#include <cassert>
#include <iostream>
#include <vector>

#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(KD_TREE_COUNT);
    search_params = flannKsTreeSearchParams(KD_TREE_SEARCH_LEAF_COUNT);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    matches.clear();
    
    cv::Mat indices;
    cv::Mat dists;
    flann_index->knnSearch(query_desc, indices, dists, k, *search_params);
    
    for (int i = 0; i < indices.size[0]; ++i) {
        matches.emplace_back();
        auto &current_result = matches.back();

        for (int j = 0; j < k; ++j) {
            current_result.emplace_back(i, indices.at<int>(i, j), dists.at<float>(i, j));
        }
    }
}
