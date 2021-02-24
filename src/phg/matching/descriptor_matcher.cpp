#include "descriptor_matcher.h"

#include <opencv2/flann/miniflann.hpp>
#include "flann_factory.h"

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    for(const auto& match : matches){
        if( match.at(0).distance < match.at(1).distance * 0.7){
            filtered_matches.push_back(match.at(0));
        }
    }

//    throw std::runtime_error("not implemented yet");
}

template<class T>
std::vector<T>  convert(const cv::Mat& m){

    assert(m.rows == 1);
    return std::vector<T>(m.begin<T>(), m.end<T>());
}

void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    const size_t  total_neighbours  = 5;  // total number of neighbours to test (including candidate)
    const size_t  consistent_matches  = 3;  // minimum number of consistent matches (including candidate)
    const float  radius_limit_scale  = 2.f;  // limit search radius by scaled median

    const int n_matches = matches.size();

    if (n_matches < total_neighbours) {
        throw std::runtime_error("DescriptorMatcher::filterMatchesClusters : too few matches");
    }

    cv::Mat points_query(n_matches, 2, CV_32FC1);
    cv::Mat points_train(n_matches, 2, CV_32FC1);
    for (int i = 0; i < n_matches; ++i) {
        points_query.at<cv::Point2f>(i) = keypoints_query[matches[i].queryIdx].pt;
        points_train.at<cv::Point2f>(i) = keypoints_train[matches[i].trainIdx].pt;
    }
//// размерность всего 2, так что точное KD-дерево
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(2);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(2);

    std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
    std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);

    // для каждой точки найти total neighbors ближайших соседей
    cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_query(n_matches, total_neighbours, CV_32FC1);
    cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_train(n_matches, total_neighbours, CV_32FC1);

    index_query->knnSearch(points_query, indices_query, distances2_query, total_neighbours, *search_params);
    index_train->knnSearch(points_train, indices_train, distances2_train, total_neighbours, *search_params);

    // оценить радиус поиска для каждой картинки
    // NB: radius2_query, radius2_train: квадраты радиуса!
    float radius2_query, radius2_train;
    {
        std::vector<double> max_dists2_query(n_matches);
        std::vector<double> max_dists2_train(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            max_dists2_query[i] = distances2_query.at<float>(i, total_neighbours - 1);
            max_dists2_train[i] = distances2_train.at<float>(i, total_neighbours - 1);
        }

        int median_pos = n_matches / 2;
        std::nth_element(max_dists2_query.begin(), max_dists2_query.begin() + median_pos, max_dists2_query.end());
        std::nth_element(max_dists2_train.begin(), max_dists2_train.begin() + median_pos, max_dists2_train.end());

        radius2_query = max_dists2_query[median_pos] * radius_limit_scale * radius_limit_scale;
        radius2_train = max_dists2_train[median_pos] * radius_limit_scale * radius_limit_scale;
    }
    //
    //    метч остается, если левое и правое множества первых total_neighbors соседей
    // в радиусах поиска(radius2_query, radius2_train) имеют как минимум consistent_matches общих элементов
    //    // TODO заполнить filtered_matches

    int consistent_matches_without_candindate = consistent_matches;

    for (int i = 0; i < matches.size(); ++i) {

        std::vector<int> query_nn = convert<int>(indices_query.row(i));
        std::vector<int> train_nn = convert<int>(indices_train.row(i));

        std::vector<float> query_dist = convert<float>(distances2_query.row(i));
        std::vector<float> train_dist = convert<float>(distances2_train.row(i));

        std::set<int> query_filter_nn, train_filter_nn;

        for (int i = 0; i < query_dist.size(); ++i) {
            if(query_dist.at(i) <radius2_query ){
                query_filter_nn.insert(query_nn.at(i));
            }
        }

        if(query_filter_nn.size() < consistent_matches_without_candindate)
            continue;

        for (int i = 0; i < train_dist.size(); ++i) {
            if(train_dist.at(i) <radius2_train ){
                train_filter_nn.insert(train_nn.at(i));
            }
        }

        if(train_filter_nn.size() < consistent_matches_without_candindate)
            continue;

        std::set<int> v_intersection;

        std::set_intersection(train_filter_nn.begin(), train_filter_nn.end(),
                              query_filter_nn.begin(), query_filter_nn.end(),
                              std::inserter(v_intersection, v_intersection.begin()));

        if(v_intersection.size() >= consistent_matches_without_candindate){
            filtered_matches.push_back(matches[i]);
        }
    }
}
