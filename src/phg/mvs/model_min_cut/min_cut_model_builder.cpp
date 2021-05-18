#include "min_cut_model_builder.h"

#include <libutils/timer.h>
#include <libutils/rasserts.h>

#include <phg/utils/point_cloud_export.h>
#include <phg/mvs/depth_maps/pm_geometry.h>

#include "min_cut_cgal_structs.h"
#include "min_cut_defines.h"
#include "min_cut_max_flow_solver.h"


struct MinCutModelBuilder::TriangulationProxy
{
    triangulation_t triangulation;
};

MinCutModelBuilder::MinCutModelBuilder()
{
    // благодаря этому трюку тяжелые CGAL инклюды не попадают во внешний хедер
    // это важно для скорости сборки и непросачивания зависимостей между модулями при использовании header-only библиотеки
    // (особенно чувствуется при инкрементальной сборке после изменения cpp файла который инклюдил бы min_cut_model_builder.h вместе со всем CGAL в придачу)
    proxy = std::shared_ptr<TriangulationProxy>(new TriangulationProxy());

    append_total_points = 0;
    append_inserted_points = 0;
    append_nn_search_total_t = 0.0;
    append_triangulation_insert_total_t = 0.0;
    append_total_t = 0.0;
}

void MinCutModelBuilder::appendToTriangulation(unsigned int camera_id, const vector3d &camera_center,
                                               const std::vector<vector3d> &points,
                                               const std::vector<double> &radiuses,
                                               const std::vector<cv::Vec3b> &colors,
                                               const std::vector<vector3d> &normals)
{
    timer total_t;

    rassert(points.size() == radiuses.size(), 23849184291241010);
    rassert(points.size() == colors.size(), 278391274120012);
    rassert(points.size() == normals.size(), 2389210130013);

    std::vector<std::pair<cgal_point_t, vertex_info_t>> points_to_insert;

    timer nn_search_t;
    for (ptrdiff_t i = 0; i < points.size(); ++i) {
        vector3d p = points[i];
        double r = radiuses[i];
        cv::Vec3b color = colors[i];
        vector3d normal = normals[i];

        bool to_merge = false; // хотим решить - надо ли очередную точку объединить с уже существующей ближайшей вершиной, или же добавить в триангуляцию как новую вершину

        vertex_handle_t nearest_vertex = proxy->triangulation.nearest_vertex(to_cgal_point(p));

        if (nearest_vertex == vertex_handle_t()) {
            // означает что мы не нашли ни одной точки, т.е. триангуляция пустая
            rassert(cameras_centers.size() == 0, 238912478912039); // такое может быть только в одном случае - когда мы первая камера, проверяем это
            to_merge = false;
        } else {
            // проверяем насколько ближайшая точка далеко
            vector3d np = from_cgal_point(nearest_vertex->point());
            float merge_threshold = r * MERGE_THRESHOLD_RADIUS_KOEF;
            if (phg::norm(np - p) < merge_threshold) {
                to_merge = true;
            } else {
                to_merge = false;
            }
        }

        vertex_info_t p_info(camera_id, color, r, p);
        if (to_merge) {
            nearest_vertex->info().merge(p_info);
            // 4001 подвиньте вершины в среднюю координату среди всех точек которые в ней зачлись
            proxy->triangulation.move(nearest_vertex, to_cgal_point(nearest_vertex->info().point));
        } else {
            points_to_insert.push_back(std::make_pair(to_cgal_point(p), p_info));
        }
    }
    append_nn_search_total_t += nn_search_t.elapsed();

    timer insertion_t;
    proxy->triangulation.insert(points_to_insert.begin(), points_to_insert.end());
    append_triangulation_insert_total_t += insertion_t.elapsed();

    rassert(cameras_centers.count(camera_id) == 0, 23781247112017);
    cameras_centers[camera_id] = camera_center;

    append_total_points += points.size();
    append_inserted_points += points_to_insert.size();

    append_total_t += total_t.elapsed();
}

void MinCutModelBuilder::printAppendStats()
{
    std::cout << "MinCutModelBuilder.append: " << append_total_points << " points processed (merged to " << to_percent(append_inserted_points, append_total_points) << "%) totally in "
    << append_total_t << " s = " << to_percent(append_nn_search_total_t, append_total_t) << "% localization + " << to_percent(append_triangulation_insert_total_t, append_total_t) << "% insertion" << std::endl;
}

namespace {
    void debugSavePointCloud(const std::string &label, const std::vector<vector3d> &points)
    {
        static int point_cloud_index = 0;
        std::string path = "data/debug/test_mesh_min_cut/debug_points/" + to_string(point_cloud_index++) + "_debug_points_" + label + ".ply";
        phg::exportPointCloud(points, path);
        std::cout << points.size() << " points exported to " << path << std::endl;
    }
}

void MinCutModelBuilder::insertBoundingBoxVertices(vector3d &bb_min, vector3d &bb_max)
{
    std::vector<vector3d> debug_points_cameras;
    std::vector<vector3d> debug_triangulation_points;
    std::vector<vector3d> debug_bounding_box_points;

    rassert(cameras_centers.size() > 0, 231278947812510105);
    bb_min = (*cameras_centers.begin()).second;
    bb_max = bb_min;
    for (auto it = cameras_centers.begin(); it != cameras_centers.end(); ++it) {
        vector3d center = (*it).second;
        for (int d = 0; d < 3; ++d) {
            bb_min[d] = std::min(bb_min[d], center[d]);
            bb_max[d] = std::max(bb_max[d], center[d]);
        }
        debug_points_cameras.push_back(center);
    }
    for (auto it = proxy->triangulation.finite_vertices_begin(); it != proxy->triangulation.finite_vertices_end(); ++it) {
        vector3d p = from_cgal_point(it->point());
        for (int d = 0; d < 3; ++d) {
            bb_min[d] = std::min(bb_min[d], p[d]);
            bb_max[d] = std::max(bb_max[d], p[d]);
        }
        debug_triangulation_points.push_back(p);
    }

    vector3d bb_size = bb_max - bb_min;
    double the_biggest_side = std::max(bb_size[0], std::max(bb_size[1], bb_size[2]));
    for (int d = 0; d < 3; ++d) {
        bb_min[d] -= 3 * the_biggest_side;
        bb_max[d] += 3 * the_biggest_side;
    }

    // вставляем в триангуляцию 3*3*3-1 точек - углы куба bb_min<->bb_max + точки на серединах его ребер и граней (но без центральной точки)
    vector3d corners[3] = {bb_min, (bb_min + bb_max) / 2.0, bb_max};

    std::vector<std::pair<cgal_point_t, vertex_info_t>> points_to_insert;
    for (int i = 0; i < 3*3*3; ++i) {
        if (i == 1+3+9) continue; // пропускаем центральную точку куба
        vector3d p;
        int powOf3 = 1;
        for (int d = 0; d < 3; ++d) {
            p[d] = corners[(i / powOf3) % 3][d];
            powOf3 *= 3;
        }
        vertex_info_t bounding_box_corner_empty_info;
        debug_bounding_box_points.push_back(p);
        points_to_insert.push_back(std::make_pair(to_cgal_point(p), bounding_box_corner_empty_info));
    }
    proxy->triangulation.insert(points_to_insert.begin(), points_to_insert.end());

    debugSavePointCloud("cameras", debug_points_cameras);
    debugSavePointCloud("triangulation", debug_triangulation_points);
    debugSavePointCloud("bounding_box", debug_bounding_box_points);
}

namespace {
    bool point_in_tetrahedron(const cell_handle_t &cell, const vector3d &point);
    std::vector<cgal_facet_t> fetchVertexBoundingFacets(const triangulation_t& triangulation, const vertex_handle_t& vi)
    {
        // давайте посмотрим на вершину, затем на все смежные с ней ячейки (тетрагедрончики), а затем возьмем все треугольники-границы этих ячеек,
        // не опирающиеся на нашу текущую точку
        // иначе говоря мы хотим построить шарик из треугольников вокруг нашей вершины, эти треугольники мы дальше будем пересекать с нашим
        // лучем трассировки, чтобы понять через какой из этих треугольников надо перешагнуть дальше в слудеющую ячейку (тетрагедрончик) триангуляции
        std::vector<cgal_facet_t> vertex_facets;

        std::vector<cell_handle_t> incident_cells;
        triangulation.incident_cells(vi, back_inserter(incident_cells));

        for (int i = 0; i < incident_cells.size(); ++i) {
            cell_handle_t cell_handle = incident_cells[i]; // одна из смежных с вершиной ячеек (т.е. тетрагедрончик)
            cgal_facet_t facet(cell_handle, cell_handle->index(vi)); // треугольник ячейки лежащий напротив нашей вершины-запроса
            rassert(!triangulation.is_infinite(facet), 2371239103129010153); // ради этой гарантии мы добавляли bounding box (чтобы не требовалось работать с бесконечными треугольниками, т.е. с опирающимися на бесконечную точку)
            vertex_facets.push_back(facet);
        }
        return vertex_facets;
    }

    // 5004 CGAL::do_intersect проверяет луч и треугольник на пересечение абсолютно точно, и это надежно, но медленно.
    // А что если мы грубо будем проверять пересечения (самописным простым кодом на float-ах)?
    bool intersect(const cgal_facet_t &facet, const vector3d &origin, const vector3d &direction) {
        vector3d p0 = from_cgal_point(facet.first->vertex((facet.second + 1) % 4)->point());
        vector3d p1 = from_cgal_point(facet.first->vertex((facet.second + 2) % 4)->point());
        vector3d p2 = from_cgal_point(facet.first->vertex((facet.second + 3) % 4)->point());

        vector3d normal = cv::normalize(((p1 - p0).cross(p2 - p1)));
        double num = (p0 - origin).dot(normal);
        double den = normal.dot(direction);
        if (std::abs(den) < 1e-8) {
            return false;
        }
        double coef = num / den;
        if (coef < 0) {
            return false;
        }
        vector3d p = origin + coef * direction;
        return normal.dot((p1 - p0).cross(p - p0)) > 0
               && normal.dot((p2 - p1).cross(p - p1)) > 0
               && normal.dot((p0 - p2).cross(p - p2)) > 0;
    }

    cgal_facet_t chooseIntersectedFacet(const triangulation_t& triangulation,
                                        const vector3d &rayFrom, const vector3d &rayTo,
                                        std::vector<cgal_facet_t> &facets, bool checkFinish=true)
    {
        // выбираем среди предложенных фэйсов (треугольников, т.е. граней ячеек) тот что пересечен нашим лучем идущим из rayFrom в rayTo
        // а так же обновляем множество фейсов (треугольников) до актуального состояния по ту сторону пересеченного фейса (по ту сторону треугольника через который мы перешагнули) 

        rassert(facets.size() > 0, 538947914120162);

        cgal_facet_t intersected_facet;

        for (int i = 0; i < facets.size(); ++i) {
            cgal_facet_t facet = facets[i];
            rassert(!triangulation.is_infinite(facet), 238192412173);

            bool my_intersection = intersect(facet, rayFrom, rayTo - rayFrom);
            if (my_intersection) {
                intersected_facet = facet;
                break;
            }
        }

        if (intersected_facet == cgal_facet_t()) {
            // если никакого пересечения не было найдено, попробуем использовать CGAL::do_intersect
            for (int i = 0; i < facets.size(); ++i) {
                cgal_facet_t facet = facets[i];
                rassert(!triangulation.is_infinite(facet), 238192412173);
                cgal_kernel_t::Ray_3 ray(to_cgal_point(rayFrom), to_cgal_point(rayTo));
                bool cgal_intersection = CGAL::do_intersect(triangulation.triangle(facet), ray);

                if (cgal_intersection) {
                    intersected_facet = facet;
                    break;
                }
            }
        }

        // надо актуализировать список треугольников-кандидатов которые должны быть проверены при следующей попытке шагнуть по лучу
        facets.clear();

        if (intersected_facet == cgal_facet_t()) {
            // если даже с  CGAL::do_intersect никакого пересечения не было найдено - возвращаем пустой фэйс (пустой треугольник)
            // вместе с опустошенным множеством треугольников для будущих проверок на трассировку луча
            return cgal_facet_t();
        }

        // проверяем - вдруг та ячейка в которую мы сейчас перешли - содержит наш пункт назначения (т.е. камеру)
        const cell_handle_t prev_cell = intersected_facet.first;
        const cell_handle_t next_cell = intersected_facet.first->neighbor(intersected_facet.second);
        rassert(!triangulation.is_infinite(prev_cell), 23891294812199);
        if (checkFinish) {
            rassert(!triangulation.is_infinite(next_cell), 23891294812200); // таким образом мы например косвенно проверяем что наш критерий остановки по ячейке содержащей камеру - срабатывает, и мы не уходим на бесконечность
            // 5003 не слишком ли часто вызывается triangulation.locate()? может оно тормозит? (поиск ячейки содержащей заданную точку)
            // заменила locate на проверку, что точка лежит внутри тетраэдра
            if (point_in_tetrahedron(next_cell, rayTo)) {
                // раз мы дошли до ячейки содержащей конец нашего пути - дальше идти не требуется, т.е. оставляем список будущих треугольников-кандидатов пустым
                return intersected_facet;
            } else if (point_in_tetrahedron(prev_cell, rayTo)) {
                // что если камера была в ячейке опирающейся на вершину-старт? т.е. мы начали свой путь из вершины и сразу оказались в ячейке с камерой (не переходя ни через один треугольник-фейс)
                // в таком случае тоже возвращаем пустой фэйс (пустой треугольник)
                // вместе с опустошенным множеством треугольников для будущих проверок на трассировку луча
                return cgal_facet_t();
            }
        }

        for (int i = 0; i < 4; ++i) {
            // добавляем в перечень треугольников-кандидатов все грани новой ячейки кроме той грани через которую мы в эту ячейку попали
            if (next_cell->neighbor(i) != prev_cell) {
                facets.push_back(cgal_facet_t(next_cell, i));
            }
        }

        return intersected_facet;
    }

    struct plane_t {
        double a;
        double b;
        double c;
        double d;

        plane_t(cgal_facet_t facet) {
            // See:
            // https://doc.cgal.org/latest/Triangulation_3/index.html
            // https://doc.cgal.org/latest/TDS_3/index.html
            //
            // The four vertices of a cell are indexed with 0, 1, 2 and 3 in positive orientation, the positive orientation 
            // being defined by the orientation of the underlying Euclidean space ℝ3 (see Figure 44.1). 
            // The neighbors of a cell are also indexed with 0, 1, 2, 3 in such a way that the neighbor indexed by i is opposite to the vertex with the same index.
            //
            // As in the underlying combinatorial triangulation (see Chapter 3D Triangulation Data Structure), edges ( 1-faces)
            // and facets ( 2-faces) are not explicitly represented: a facet is given by a cell and an index (the facet i of a 
            // cell c is the facet of c that is opposite to the vertex with index i) and an edge is given by a cell 
            // and two indices (the edge (i,j) of a cell c is the edge whose endpoints are the vertices of c with indices i and j). See Figure 45.1. 
            vector3d p0 = from_cgal_point(facet.first->vertex((facet.second + 1) % 4)->point());
            vector3d p1 = from_cgal_point(facet.first->vertex((facet.second + 2) % 4)->point());
            vector3d p2 = from_cgal_point(facet.first->vertex((facet.second + 3) % 4)->point());

            vector3d normal = cv::normalize(((p1 - p0).cross(p2 - p1)));
            a = normal[0];
            b = normal[1];
            c = normal[2];
            d = - (a * p0[0] + b * p0[1] + c * p0[2]);
            rassert(!isDegenerate(), 237812321230);
        }


        double distance(const vector3d &p) {
            return fabs(a * p[0] + b * p[1] + c * p[2] + d) / sqrt(a * a + b * b + c * c);
        }

        vector3d normal() {
            return vector3d(a, b, c);
        }

        bool isDegenerate() {
            return a == 0.0 && b == 0.0 && c == 0.0;
        }

        double distanceToIntersection(const vector3d &from, const vector3d &ray) {
            double angle_cos = fabs(cv::normalize(ray).dot(normal()));
            if (angle_cos < 1e-5) {
                // плоскость и луч почти параллельны, вычисления ненадежны
                return -1.0;
            }
            rassert(angle_cos != 0.0 && !isDegenerate(), 849284912247);
            double height_distance = distance(from);
            double ray_distance = height_distance / angle_cos;
            return ray_distance;
        }
    };

    double evaluate(plane_t plane, vector3d point) {
        return plane.a * point[0] + plane.b * point[1] + plane.c * point[2] + plane.d;
    }

    bool same_side(const vector3d &point, const vector3d &vertex, cgal_facet_t facet) {
        plane_t plane(facet);
        return evaluate(plane, point) * evaluate(plane, vertex) > 0;
    }

    bool point_in_tetrahedron(const cell_handle_t &cell, const vector3d &point) {
        for (int i = 0; i < 4; ++i) {
            cgal_facet_t facet(cell, i);
            if (!same_side(point, from_cgal_point(cell->vertex(i)->point()), facet))
                return false;
        }
        return true;
    }
}

void MinCutModelBuilder::buildMesh(std::vector<cv::Vec3i> &mesh_faces, std::vector<vector3d> &mesh_vertices, std::vector<cv::Vec3b> &mesh_vertices_color)
{
    timer total_t;

    // обрамляем наше облако точек в bounding box, чтобы при трассировке лучей не пришлось разбираться с бесконечно удаленной точкой
    // (которая очень красива конечно, но с ней пришлось бы отдельно пересекать луч и бесконечный треугольник)
    vector3d bb_min, bb_max;
    insertBoundingBoxVertices(bb_min, bb_max);

    size_t ncells = 0;
    for (auto ci = proxy->triangulation.all_cells_begin(); ci != proxy->triangulation.all_cells_end(); ++ci) {
        ci->info().cell_id = ncells++;
        ci->info().t_capacity = 0.0;
        ci->info().s_capacity = 0.0;
        for (int i = 0; i < 4; ++i) {
            ci->info().facets_capacities[i] = 0.0;
        }
    }

    timer rays_traversing_t;
    double avg_triangles_intersected_per_ray = 0;
    size_t nrays = 0;
    size_t lost_rays = 0;

    for (auto vi = proxy->triangulation.all_vertices_begin(); vi != proxy->triangulation.all_vertices_end(); ++vi) {
        if (vi->info().camera_ids.size() == 0) {
            // 2004 подумайте и напишите тут какие вершины бывают без камер вообще? почему мы их пропускаем? что и почему случится если убрать это пропускание?
            // для вершин bounding box camera_ids.size() == 0
            // эти вершины не являются внутренними и построить для них "шарик" из треугольников вокруг не получится
            continue;
        }

        const vector3d point0 = from_cgal_point(vi->point());
        const float r = vi->info().radius;
        const std::vector<cgal_facet_t> facets_around_point0 = fetchVertexBoundingFacets(proxy->triangulation, vi);

        for (unsigned int ci = 0; ci < vi->info().camera_ids.size(); ++ci) {
            // для каждой вершины триангуляции point0 и каждой камеры к которой эта точка имеет отношение (т.е. содержится где-то в карте глубины)

            unsigned int camera_key = vi->info().camera_ids[ci];
            rassert(cameras_centers.count(camera_key) > 0, 238123812938191822);
            const vector3d camera_center = cameras_centers[camera_key];
            const vector3d ray_from_camera = cv::normalize(point0 - camera_center);
            const vector3d ray_to_camera = cv::normalize(camera_center - point0);
            const double distance_to_camera = phg::norm(camera_center - point0);

            {
                // хотим найти ячейку триангуляции лежащую внутри поверхности (т.е. сразу за этим лучем видимости camera_center->point0):
                // давайте посмотрим на вершину, затем на все смежные с ней ячейки (тетрагедрончики), а затем возьмем все треугольники-границы этих ячеек (facets_around_point0),
                // не опирающиеся на нашу текущую точку,
                // иначе говоря мы смотрим на шарик из треугольников вокруг нашей вершины
                // и теперь среди этих треугольников мы ищем тот что пересекается лучем идущим глубже за точку на поверхности
                // этот треугольник - это грань искомой ячейки триангуляции внутри поверхности

                // 3002 сделайте соединение со стоком в ячейке не сразу за вершиной, а на небольшом углублении (пропорционально размеру точки)
                const vector3d point1 = point0 + 3 * r * ray_from_camera;
                const cell_handle_t last_facet = proxy->triangulation.locate(to_cgal_point(point1));

                std::vector<cgal_facet_t> cur_facets = facets_around_point0;
                cgal_facet_t intersected_facet = chooseIntersectedFacet(proxy->triangulation, point0, point1, cur_facets, false);
                rassert(intersected_facet != cgal_facet_t(), 2378213120305);

                double prev_distance = 0.0;
                while (intersected_facet.first != last_facet && prev_distance < 3 * r) {
                    // посчитаем какой путь мы уже прошли от точки, для этого надо найти расстояние от точки до места пересечения луча и треугольника (т.е. плоскости на которой он лежит, т.к. мы уже знаем что треугольник мы пересекаем лучем)
                    plane_t facet_plane(intersected_facet);
                    double distance_from_surface = facet_plane.distanceToIntersection(point0, ray_from_camera);
                    if (distance_from_surface < 0.0 || distance_from_surface < prev_distance * 0.99) {
                        // плоскость и луч почти параллельны, вычисления ненадежны, расстояние до пересечения может быть странным (например монотонность может сломаться)
                        // в таком случае оставим предыдущую оценку пройденного пути
                        break;
                    }
                    prev_distance = distance_from_surface;

                    // 3001 сделайте пропускные способности на ребрах не единичными а затухающими тем сильнее чем ближе к поверхности
                    float weight = LAMBDA_IN * (1 - std::exp(-distance_from_surface * distance_from_surface / (2 * r * r)));

                    // увеличиваем пропускную способность на треугольнике-ребре (в направлении от камеры к точке)
                    intersected_facet.first->info().facets_capacities[intersected_facet.second] += weight;
                    cgal_facet_t next_facet = chooseIntersectedFacet(proxy->triangulation, point0, point1, cur_facets, false);
                    if (next_facet == cgal_facet_t()) {
                        break;
                    }
                    intersected_facet = next_facet;
                }

                // добавляем пропускной способности из этой ячейки (из этого тетрагедрончика) к стоку
                intersected_facet.first->info().t_capacity += LAMBDA_IN;
            }
            
            // шагаем от точки до камеры выставляя веса на треугольниках (они же ребра в графе) которые пересекаются по мере трассировки луча
            std::vector<cgal_facet_t> cur_facets = facets_around_point0; // это актуальные на данный момент треугольники-кандидаты для пересечения с лучем

            double prev_distance = 0.0;
            size_t steps = 0;
            while (cur_facets.size() > 0) {
                const cgal_facet_t intersected_facet = chooseIntersectedFacet(proxy->triangulation, point0, camera_center, cur_facets);
                if (intersected_facet == cgal_facet_t() && steps == 0) {
                    // единственное исключение - это когда отрезок из вершины до камеры не пересекает ни одного треугольника (т.е. когда камера находится в смежной с вершиной ячейке)
                    break;
                } else if (intersected_facet == cgal_facet_t()) {
                    ++lost_rays;
                    break;
                }
                rassert(!proxy->triangulation.is_infinite(intersected_facet), 238921410312); // если треугольник бесконечный - с ним сложно работать, чтобы такого не случалось - мы добавили фиктивные точки - создали bounding box в insertBoundingBoxVertices()
                ++steps;

                // отзеркаливаем треугольник (грань ячейки), т.к. нам нужно обновить пропускную способность ребра по направлению от камеры к точке, а перешагивали по треугольникам мы по направлению от точки к камере
                const cgal_facet_t mirrored_intersected_facet = proxy->triangulation.mirror_facet(intersected_facet);

                // находим две ячейки (находящиеся по разные стороны от только что пересеченного треугольника):
                // следующая всмысле шагания ячейка (та что ближе к камере)
                const cell_handle_t next_cell = mirrored_intersected_facet.first;
                const int next_cell_facet_subindex = mirrored_intersected_facet.second;
                rassert(next_cell_facet_subindex >= 0 && next_cell_facet_subindex < 4, 238129481240292);
                // предыдущая всмысле шагания ячейка (та что ближе к точке)
                const cell_handle_t prev_cell = next_cell->neighbor(next_cell_facet_subindex);

                // посчитаем какой путь мы уже прошли от точки, для этого надо найти расстояние от точки до места пересечения луча и треугольника (т.е. плоскости на которой он лежит, т.к. мы уже знаем что треугольник мы пересекаем лучем)
                plane_t facet_plane(intersected_facet);
                double distance_from_surface = facet_plane.distanceToIntersection(point0, ray_to_camera);
                if (distance_from_surface < 0.0 || distance_from_surface < prev_distance * 0.99) {
                    // плоскость и луч почти параллельны, вычисления ненадежны, расстояние до пересечения может быть странным (например монотонность может сломаться)
                    // в таком случае оставим предыдущую оценку пройденного пути
                    distance_from_surface = prev_distance;
                    // после добавления своего вычисления пересечения столкнулась с зацикливанием, поэтому теперь когда плоскость и луч почти параллельны, считаю, что луч потерян
                    ++lost_rays;
                    break;
                } else {
                    rassert(distance_from_surface > prev_distance * 0.99, 23789247124210293); // дополнительная проверка на разумность происходящего, мы удаляемся от точки - приближаемся к камере
                    rassert(distance_from_surface < distance_to_camera * 1.01, 238712973120335); // проверяем что chooseIntersectedFacet справился со своей задачей "остановиться когда мы дойдем до ячейки содержащей камеру"
                }
                prev_distance = distance_from_surface;

                // 3001 сделайте пропускные способности на ребрах не единичными а затухающими тем сильнее чем ближе к поверхности
                float weight = LAMBDA_OUT * (1 - std::exp(-distance_from_surface * distance_from_surface / (2 * r * r)));

                // увеличиваем пропускную способность на треугольнике-ребре (в направлении от камеры к точке)
                next_cell->info().facets_capacities[next_cell_facet_subindex] += weight;

                if (cur_facets.size() == 0) {
                    // если на будущее у нас нет кандидатов-треугольников, значит мы закончили наш путь и следующая ячейка содержит нашу камеру
                    rassert(next_cell == proxy->triangulation.locate(to_cgal_point(camera_center)), 238791248120328); // проверяем это
                    // добавляем пропускной способности из истока к ячейке с камерой (к тетрагедрончику содержащему точку центра камеры)
                    next_cell->info().s_capacity += LAMBDA_IN;
                    // 2005 изменится ли что-то если сильно увеличить пропускные способности ребер от истока? (т.е. сделать пропускную способность из истока равной бесконечности?)
                    // не изменится, так как увеличесние пропускной способности может только повлиять на то, что ребро не попадет в минимальный разрез, но ребра от истока в нем и так не содержатся
                }
            }
            avg_triangles_intersected_per_ray += steps;
            ++nrays;
        }
    }
    double rays_traversed_time = rays_traversing_t.elapsed();
    if (nrays > 0) avg_triangles_intersected_per_ray /= nrays;
    std::cout << "Visibility rays traversed in " << rays_traversed_time << " s: " << nrays - lost_rays << "/" << nrays << " rays with " << avg_triangles_intersected_per_ray << " avg triangle intersections" << std::endl;

    MaxFlow<size_t> max_flow_solver(ncells);

    timer graph_construction_t;
    for (auto ci = proxy->triangulation.all_cells_begin(); ci != proxy->triangulation.all_cells_end(); ++ci) {
        size_t cell_id = ci->info().cell_id;

        float s = ci->info().s_capacity;
        float t = ci->info().t_capacity;

        max_flow_solver.addNode(cell_id, s, t);
    }
    for (auto ci = proxy->triangulation.all_cells_begin(); ci != proxy->triangulation.all_cells_end(); ++ci) {
        size_t a = ci->info().cell_id;
        for (int i = 0; i < 4; ++i) {
            float capacity = ci->info().facets_capacities[i];
    
            cell_handle_t cj = ci->neighbor(i);
            int j = cj->index(ci);
    
            size_t b = cj->info().cell_id;
    
            float reverseCapacity = cj->info().facets_capacities[j];

            if (a < b) {
                max_flow_solver.addEdge(a, b, capacity, reverseCapacity);
            }
        }
    }
    double graph_construction_time = graph_construction_t.elapsed();
    std::cout << "Graph constructed in " << graph_construction_time << " s" << std::endl;

    timer min_cut_t;
    float max_flow = max_flow_solver.computeMaxFlow();
    double min_cut_solver_time = min_cut_t.elapsed();
    std::cout << "Max flow solver processed in " << min_cut_solver_time << " s (max flow: " << max_flow << ")" << std::endl;

    const size_t VERTEX_NOT_ON_SURFACE_RESULT = std::numeric_limits<size_t>::max();
    for (auto vi = proxy->triangulation.all_vertices_begin(); vi != proxy->triangulation.all_vertices_end(); ++vi) {
        vi->info().vertex_on_surface_id = VERTEX_NOT_ON_SURFACE_RESULT;
    }

    mesh_faces.clear();

    std::vector<vector3d> debug_surface_points;
    std::vector<vector3d> debug_non_surface_points;

    timer surface_extraction_t;
    size_t mesh_nvertices = 0;
    for (auto ci = proxy->triangulation.all_cells_begin(); ci != proxy->triangulation.all_cells_end(); ++ci) {
        size_t a = ci->info().cell_id;
        for (int i = 0; i < 4; ++i) {
            if (proxy->triangulation.is_infinite(ci, i))
                continue;

            cell_handle_t cj = ci->neighbor(i);
            int j = cj->index(ci);

            size_t b = cj->info().cell_id;

            bool a_inside = !max_flow_solver.isNodeOnSrcSide(a);
            bool b_outside = max_flow_solver.isNodeOnSrcSide(b);
            if (!a_inside || !b_outside) {
                continue;
            }
            rassert(a_inside && b_outside, 23892183120391);

            cv::Vec3i face;

            // иначе говоря сделайте так чтобы такие треугольники не добавлялись в результирующую модель эти большие красные треугольники
            bool is_fictitious = false;
            for (int v_index = 1; v_index <= 3; ++v_index) {
                auto vi = ci->vertex((i + v_index) % 4);
                vector3d p = from_cgal_point(vi->point());
                for (int k = 0; k < 3; ++k) {
                    if (p[k] == bb_min[k] || p[k] == bb_max[k] || p[k] == 0.5 * (bb_min[k] + bb_max[k])) {
                        is_fictitious = true;
                    }
                }
                size_t& surface_vertex_id = vi->info().vertex_on_surface_id;
                if (surface_vertex_id == VERTEX_NOT_ON_SURFACE_RESULT) {
                    surface_vertex_id = mesh_nvertices++;
                }
                face[v_index - 1] = surface_vertex_id;
            }
            if (is_fictitious) {
                continue;
            }

            // иначе говоря оказывается что порядок обхода вершин в треугольнике не всегда корректен
            // подумайте чем это вызывано и поправьте (лучше всего это делать посматривая на картинку 'Figure 44.1' в документации https://doc.cgal.org/latest/Triangulation_3/index.html )
            if (i % 2 == 1) {
                std::swap(face[0], face[1]);
            }
            mesh_faces.push_back(face);
        }
    }

    mesh_vertices.resize(mesh_nvertices);
    mesh_vertices_color.resize(mesh_nvertices);
    for (auto vi = proxy->triangulation.all_vertices_begin(); vi != proxy->triangulation.all_vertices_end(); ++vi) {
        size_t surface_vertex_id = vi->info().vertex_on_surface_id;
        if (surface_vertex_id == VERTEX_NOT_ON_SURFACE_RESULT) {
            debug_non_surface_points.push_back(from_cgal_point(vi->point()));
            continue;
        } else {
            debug_surface_points.push_back(from_cgal_point(vi->point()));
        }

        mesh_vertices[surface_vertex_id] = from_cgal_point(vi->point());
        for (int i = 0; i < 3; ++i) {
            rassert(mesh_vertices[surface_vertex_id][i] >= bb_min[i], 23782310488);
            rassert(mesh_vertices[surface_vertex_id][i] <= bb_max[i], 23782310489);
        }
        mesh_vertices_color[surface_vertex_id] = vi->info().color;
        std::swap(mesh_vertices_color[surface_vertex_id][0], mesh_vertices_color[surface_vertex_id][2]); // BGR -> RGB
    }
    for (size_t i = 0; i < mesh_faces.size(); ++i) {
        for (size_t j = 0; j < 3; ++j) {
            rassert(mesh_faces[i][j] >= 0 && mesh_faces[i][j] < mesh_nvertices, 2378127310434);
        }
    }
    double surface_extraction_time = surface_extraction_t.elapsed();

    double total_time = total_t.elapsed();
    std::cout << "MinCutModelBuilder.buildMesh: graph with " << proxy->triangulation.number_of_vertices() << " vertices processed in "
              << total_time << " s = " << to_percent(rays_traversed_time, total_time) << "% rays tracing + " << to_percent(graph_construction_time, total_time) << "% graph construction + " << to_percent(min_cut_solver_time, total_time) << "% min-cut + " << to_percent(surface_extraction_time, total_time) << "% mesh extraction" << std::endl;
    std::cout << "Resulting mesh: " << mesh_nvertices << " vertices, " << mesh_faces.size() << " faces" << std::endl;

    debugSavePointCloud("surface", debug_surface_points);
    debugSavePointCloud("non_surface", debug_non_surface_points);
}
// TODO 3500 Weak support: реализуйте идею из jancosek2011 - Multi-View Reconstruction Preserving Weakly-Supported Surfaces - https://compsciclub.ru/attachments/classes/file_XyLpDjLx/jancosek2011.pdf