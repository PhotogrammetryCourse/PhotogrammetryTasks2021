#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <libutils/timer.h>
#include <libutils/rasserts.h>
#include <libutils/string_utils.h>

#include <phg/utils/point_cloud_export.h>
#include <phg/utils/cameras_bundler_import.h>
#include <phg/mvs/depth_maps/pm_depth_maps.h>
#include <phg/mvs/depth_maps/pm_geometry.h>

#include "utils/test_utils.h"

//________________________________________________________________________________
// Datasets:

// достаточно чтобы у вас работало на этом датасете, тестирование на Travis CI тоже ведется на нем
//#define DATASET_DIR                  "saharov32"
//#define DATASET_DOWNSCALE            4

//#define DATASET_DIR                  "temple47"
//#define DATASET_DOWNSCALE            2

// скачайте картинки этого датасета в папку data/src/datasets/herzjesu25/ по ссылке из файла LINK.txt в папке датасета
#define DATASET_DIR                  "herzjesu25"
#define DATASET_DOWNSCALE            8
//________________________________________________________________________________


class Dataset {
public:
    Dataset() : ncameras(0), calibration(0, 0)
    {}

    size_t                                  ncameras;

    std::vector<cv::Mat>                    cameras_imgs;
    std::vector<cv::Mat>                    cameras_imgs_grey;
    std::vector<std::string>                cameras_labels;
    std::vector<matrix34d>                  cameras_P;
    std::vector<std::vector<cv::KeyPoint>>  cameras_keypoints;

    std::vector<float>                      cameras_depth_min;
    std::vector<float>                      cameras_depth_max;

    phg::Calibration                        calibration;

    std::vector<phg::Track>                 tracks;
    std::vector<vector3d>                   tie_points;
};


Dataset loadDataset()
{
    timer t;

    Dataset dataset;

    std::string images_list_filename = std::string("data/src/datasets/") + DATASET_DIR + "/ordered_filenames.txt";
    std::ifstream in(images_list_filename);
    if (!in) {
        throw std::runtime_error("Can't read file: " + to_string(images_list_filename)); // проверьте 'Working directory' в 'Edit Configurations...' в CLion (должна быть корневая папка проекта, чтобы относительные пути к датасетам сталик орректны)
    }
    in >> dataset.ncameras;

    std::cout << "loading " << dataset.ncameras << " images..." << std::endl;
    for (size_t ci = 0; ci < dataset.ncameras; ++ci) {
        std::string img_name;
        in >> img_name;
        std::string img_path = std::string("data/src/datasets/") + DATASET_DIR + "/" + img_name;
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION); // чтобы если камера записала в exif-tag повернута она была или нет - мы получили сырую картинку, без поворота с учетом этой информации, ведь одну и ту же камеру могли повернуть по-разному (напр. saharov32)

        if (img.empty()) {
            throw std::runtime_error("Can't read image: " + to_string(img_path));
        }

        // выполняем опциональное уменьшение картинки
        int downscale = DATASET_DOWNSCALE;
        while (downscale > 1) {
            cv::pyrDown(img, img);
            rassert(downscale % 2 == 0, 1249219412940115);
            downscale /= 2;
        }

        if (ci == 0) {
            dataset.calibration.width_ = img.cols;
            dataset.calibration.height_ = img.rows;
            std::cout << "resolution: " << img.cols << "x" << img.rows << std::endl;
        } else {
            rassert(dataset.calibration.width_  == img.cols, 2931924190089);
            rassert(dataset.calibration.height_ == img.rows, 2931924190090);
        }

        cv::Mat grey;
        cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);

        dataset.cameras_imgs.push_back(img);
        dataset.cameras_imgs_grey.push_back(grey);
        dataset.cameras_labels.push_back(img_name);
    }

    phg::importCameras(std::string("data/src/datasets/") + DATASET_DIR + "/cameras.out",
                       dataset.cameras_P, dataset.calibration, dataset.tie_points, dataset.tracks, dataset.cameras_keypoints, DATASET_DOWNSCALE);

    dataset.cameras_depth_max.resize(dataset.ncameras);
    dataset.cameras_depth_min.resize(dataset.ncameras);
    #pragma omp parallel for schedule(dynamic, 1)
    for (ptrdiff_t ci = 0; ci < dataset.ncameras; ++ci) {
        double depth_min = std::numeric_limits<double>::max();
        double depth_max = 0.0;

        for (size_t ti = 0; ti < dataset.tracks.size(); ++ti) {
            ptrdiff_t kpt = -1;
            auto img_kpt_pairs = dataset.tracks[ti].img_kpt_pairs;
            for (size_t i = 0; i < img_kpt_pairs.size(); ++i) {
                if (img_kpt_pairs[i].first == ci) {
                    kpt = img_kpt_pairs[i].second;
                }
            }
            if (kpt == -1)
                continue; // эта ключевая точка не имеет отношения к текущей камере ci

            vector3d tie_point = dataset.tie_points[ti];
                
            vector3d px = phg::project(tie_point, dataset.calibration, dataset.cameras_P[ci]);

            // проверяем project->unproject на идемпотентность
            // при отладке удобно у отлаживаемого цикла закомментировать #pragma omp parallel for
            // еще можно наспамить много project-unproject вызвов строчка за строчкой, чтобы при отладке не перезапускать программу
            // а просто раз за разом просматривать как проходит исполнение этих функций до понимания что пошло не так
            vector3d point_test = phg::unproject(px, dataset.calibration, phg::invP(dataset.cameras_P[ci]));

            vector3d diff = tie_point - point_test;
            double norm2 = phg::norm2(diff);
            rassert(norm2 < 0.0001, 241782412410125);

            double depth = px[2];
            rassert(depth > 0.0, 238419481290132);
            depth_min = std::min(depth_min, depth);
            depth_max = std::max(depth_max, depth);
        }

        // имеет смысл расширить диапазон глубины, т.к. ключевые точки по которым он построен - лишь ориентир
        double depth_range = depth_max - depth_min;
        depth_min = std::max(depth_min - 0.25 * depth_range, depth_min / 2.0);
        depth_max =          depth_max + 0.25 * depth_range;

        rassert(depth_min > 0.0, 2314512515210146);
        rassert(depth_min < depth_max, 23198129410137);

        dataset.cameras_depth_max[ci] = depth_max;
        dataset.cameras_depth_min[ci] = depth_min;
    } 

    std::cout << DATASET_DIR << " dataset loaded in " << t.elapsed() << " s" << std::endl;

    std::vector<vector3d> tie_points_and_cameras;
    std::vector<cv::Vec3b> points_colors;
    generateTiePointsCloud(dataset.tie_points, dataset.tracks, dataset.cameras_keypoints, dataset.cameras_imgs, std::vector<char>(dataset.ncameras, true), dataset.cameras_P, dataset.ncameras,
                           tie_points_and_cameras, points_colors);

    std::string tie_points_filename = std::string("data/debug/test_depth_maps_pm/") + getTestName() + "/0_tie_points_and_cameras" + to_string(dataset.ncameras) + ".ply";
    phg::exportPointCloud(tie_points_and_cameras, tie_points_filename, points_colors);
    std::cout << "tie points cloud with cameras exported to: " << tie_points_filename << std::endl;

    return dataset;
}


TEST (DepthMap, FirstStereoPair) {
    Dataset dataset = loadDataset();
    phg::PMDepthMapsBuilder builder(dataset.ncameras, dataset.cameras_imgs, dataset.cameras_imgs_grey, dataset.cameras_labels, dataset.cameras_P, dataset.calibration);
    
    size_t ci = 2;
    size_t cameras_limit = 5;

    dataset.ncameras = cameras_limit;
    cv::Mat depth_map, normal_map, cost_map;
    builder.buildDepthMap(ci, depth_map, cost_map, normal_map, dataset.cameras_depth_min[ci], dataset.cameras_depth_max[ci]);
}
