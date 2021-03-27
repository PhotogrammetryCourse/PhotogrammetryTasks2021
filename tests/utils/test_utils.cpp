#include "test_utils.h"

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

#include <libutils/rasserts.h>

#include <phg/sfm/ematrix.h>


cv::Mat concatenateImagesLeftRight(const cv::Mat &img0, const cv::Mat &img1) {
    // это способ гарантировать себе что предположение которое явно в этой функции есть (совпадение типов картинок)
    // однажды не нарушится (по мере изменения кода) и не приведет к непредсказуемым последствиям
    // в отличие от assert() у таких rassert есть три преимущества:
    // 1) они попадают в т.ч. в релизную сборку
    // 2) есть (псевдо)уникальный идентификатор по которому легко найти где это произошло
    //    (в отличие от просто __LINE__, т.к. даже если исходный файл угадать и легко, то нумерация строк может меняться от коммита к коммиту,
    //     а падение могло случится у пользователя на старорй версии)
    // 3) есть общая удобная точка остановки на которую легко поставить breakpoint - rasserts.cpp/debugPoint()
    rassert(img0.type() == img1.type(), 125121612363131);
    rassert(img0.channels() == img1.channels(), 136161251414);

    size_t width = img0.cols + img1.cols;
    size_t height = std::max(img0.rows, img1.rows);

    cv::Mat res(height, width, img0.type());
    img0.copyTo(res(cv::Rect(0, 0, img0.cols, img0.rows)));
    img1.copyTo(res(cv::Rect(img0.cols, 0, img1.cols, img1.rows)));

    return res;
}


std::string getTestName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}


std::string getTestSuiteName() {
    return ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name();
}

void drawMatches(const cv::Mat &img1,
                 const cv::Mat &img2,
                 const std::vector<cv::KeyPoint> &keypoints1,
                 const std::vector<cv::KeyPoint> &keypoints2,
                 const std::vector<cv::DMatch> &matches,
                 const std::string &path)
{
    cv::Mat img_matches;
    drawMatches( img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1),
                 cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::imwrite(path, img_matches);
}

void generateTiePointsCloud(const std::vector<vector3d> &tie_points,
                            const std::vector<phg::Track> &tracks,
                            const std::vector<std::vector<cv::KeyPoint>> &keypoints,
                            const std::vector<cv::Mat> &imgs,
                            const std::vector<char> &aligned,
                            const std::vector<matrix34d> &cameras,
                            int ncameras,
                            std::vector<vector3d> &tie_points_and_cameras,
                            std::vector<cv::Vec3b> &tie_points_colors)
{
    rassert(tie_points.size() == tracks.size(), 24152151251241);

    tie_points_and_cameras.clear();
    tie_points_colors.clear();

    for (int i = 0; i < (int) tie_points.size(); ++i) {
        const phg::Track &track = tracks[i];
        if (track.disabled)
            continue;

        int img = track.img_kpt_pairs.front().first;
        int kpt = track.img_kpt_pairs.front().second;
        cv::Vec2f px = keypoints[img][kpt].pt;
        tie_points_and_cameras.push_back(tie_points[i]);
        tie_points_colors.push_back(imgs[img].at<cv::Vec3b>(px[1], px[0]));
    }

    for (int i_camera = 0; i_camera < ncameras; ++i_camera) {
        if (!aligned[i_camera]) {
            throw std::runtime_error("camera " + std::to_string(i_camera) + " is not aligned");
        }

        matrix3d R;
        vector3d O;
        phg::decomposeUndistortedPMatrix(R, O, cameras[i_camera]);

        tie_points_and_cameras.push_back(O);
        tie_points_colors.push_back(cv::Vec3b(0, 0, 255));
        tie_points_and_cameras.push_back(O + R.t() * cv::Vec3d(0, 0, 1));
        tie_points_colors.push_back(cv::Vec3b(255, 0, 0));
    }
}
