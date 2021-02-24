#include <iostream>
#include "flann_matcher.h"
#include "flann_factory.h"


phg::FlannMatcher::FlannMatcher()
{
    // параметры для приближенного поиска
    index_params = flannKdTreeIndexParams(6);
    search_params = flannKsTreeSearchParams(32);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    flann_index = flannKdTreeIndex(train_desc, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc, std::vector<std::vector<cv::DMatch>> &matches, int k) const
{
    using namespace std;

    //     cout << query_desc << endl;

    size_t n_desc = query_desc.rows;
    cv::Mat indices(n_desc, k, CV_32SC1);
    cv::Mat dists(n_desc, k, CV_32FC1);

    //indices - список индексов точек из train_desc
    flann_index->knnSearch(query_desc, indices, dists, k, *(search_params.get() ));

    matches.clear();
    matches.resize(indices.rows);
    for (int i = 0; i < indices.rows; ++i) {
        for (int j = 0; j < indices.cols; ++j) {
            int train_idx = indices.at<int>(i,j);
            float distance = std::sqrt(dists.at<float>(i,j));
            int query_idx = i;
            cv::DMatch match(query_idx,train_idx,distance);
            matches.at(i).push_back(match);
        }
    }
}
