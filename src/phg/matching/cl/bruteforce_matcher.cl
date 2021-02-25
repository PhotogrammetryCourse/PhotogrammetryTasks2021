//#ifndef RUN
//
//#include <libgpu/opencl/cl/clion_defines.cl>
//#define KEYPOINTS_PER_WG  4
//#define __attribute__(args)
//
//#endif

#define NDIM 128 // размерность дескриптора, мы полагаемся на то что она совпадает с размером нашей рабочей группы

__attribute__((reqd_work_group_size(NDIM, 1, 1)))
__kernel void bruteforce_matcher(__global const float* train,
                                 __global const float* query,
                                 __global        uint* res_train_idx,
                                 __global        uint* res_query_idx,
                                 __global       float* res_distance,
                                 uint n_train_desc,
                                 uint n_query_desc)
{
    // каждая рабочая группа обрабатывает KEYPOINTS_PER_WG=4 дескриптора из query (сопоставляет их со всеми train)

    const uint dim_id = get_global_id(0); // от 0 до 127, номер размерности за которую ответственен поток
    const uint query_id0 = KEYPOINTS_PER_WG * get_global_id(1); // номер первого дескриптора из четверки запросов query, которые наша рабочая группа должна сопоставлять

    // храним KEYPOINTS_PER_WG=4 дескриптора-query:
    __local float query_local[KEYPOINTS_PER_WG * NDIM];
    // храним два лучших сопоставления для каждого дескриптора-query:
    __local uint  res_train_idx_local[KEYPOINTS_PER_WG * 2];
    __local float res_distance2_local[KEYPOINTS_PER_WG * 2]; // храним квадраты чтобы не считать корень до самого последнего момента
    // заполняем текущие лучшие дистанции большими значениями
    if (dim_id < KEYPOINTS_PER_WG * 2) {
        res_distance2_local[dim_id] = FLT_MAX; // полагаемся на то что res_distance2_local размера KEYPOINTS_PER_WG*2==4*2<dim_id<=NDIM==128
    }

    // грузим 4 дескриптора-query (для каждого из четырех дескрипторов каждый поток грузит значение своей размерности dim_id)
    for (int desc = 0; desc < KEYPOINTS_PER_WG; ++desc) {
        query_local[desc * NDIM + dim_id] = query_id0 + desc < n_query_desc ? query[(query_id0 + desc) * NDIM + dim_id] : 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE); // дожидаемся прогрузки наших дескрипторов-запросов


    __local float dist2_for_reduction[NDIM];
    for (int train_idx = 0; train_idx < n_train_desc; ++train_idx) {
        float train_value_dim = train[train_idx * NDIM + dim_id];
        for (int query_local_i = 0; query_local_i < KEYPOINTS_PER_WG; ++query_local_i) {
            float query_value_dim = query_local[query_local_i * NDIM + dim_id];
            dist2_for_reduction[dim_id] = pow((query_value_dim - train_value_dim), 2); // * (query_value_dim - train_value_dim);
            // посчитать квадрат расстояния по нашей размерности (dim_id) и сохранить его в нашу ячейку в dist2_for_reduction

            barrier(CLK_LOCAL_MEM_FENCE);
            // суммируем редукцией все что есть в dist2_for_reduction
            int step = NDIM / 2;
            while (step > 0) {
                if (dim_id < step) {
                    float a = dist2_for_reduction[dim_id];
                    float b = dist2_for_reduction[dim_id + step];
                    dist2_for_reduction[dim_id] = a + b;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                step /= 2;
            }

            if (dim_id == 0) {
                // master поток смотрит на полученное расстояние и проверяет не лучше ли оно чем то что было до сих пор
                float dist2 = dist2_for_reduction[0]; // взяли найденную сумму квадратов (это квадрат расстояния до текущего кандидата train_idx)

                #define BEST_INDEX        0
                #define SECOND_BEST_INDEX 1
                const int bestidx = query_local_i * 2 + BEST_INDEX;
                const int sndbestidx = query_local_i * 2 + SECOND_BEST_INDEX;
                // пытаемся улучшить самое лучшее сопоставление для локального дескриптора
                if (dist2 <= res_distance2_local[bestidx]) {
                    // не забываем что прошлое лучшее сопоставление теперь стало вторым по лучшевизне (на данный момент)
                    res_distance2_local[sndbestidx] = res_distance2_local[bestidx];
                    res_train_idx_local[sndbestidx] = res_train_idx_local[bestidx];
                    // заменяем нашим (dist2, train_idx) самое лучшее сопоставление для локального дескриптора
                    res_distance2_local[bestidx] = dist2;
                    res_train_idx_local[bestidx] = train_idx;

                } else if (dist2 <= res_distance2_local[sndbestidx]) { // может мы улучшили хотя бы второе по лучшевизне сопоставление?
                    // заменяем второе по лучшевизне сопоставление для локального дескриптора
                    res_distance2_local[sndbestidx] = dist2;
                    res_train_idx_local[sndbestidx] = train_idx;
                }
            }
        }
    }

    // итак, мы нашли два лучших сопоставления для наших KEYPOINTS_PER_WG дескрипторов, надо сохрнить эти результаты в глобальную память
    if (dim_id < KEYPOINTS_PER_WG * 2) { // полагаемся на то что нам надо прогрузить KEYPOINTS_PER_WG*2==4*2<dim_id<=NDIM==128
        const int query_local_i = dim_id / 2;
        const int k = dim_id % 2;
        const int query_id = query_id0 + query_local_i;
        const int loc_idx = query_local_i * 2 + k;
        const int glob = query_id * 2 + k;

        if (query_id < n_query_desc) {
            res_train_idx[glob] = res_train_idx_local[loc_idx];
            res_query_idx[glob] = query_id;//  хм, не масло масленное ли? :))))
            res_distance[glob] = sqrt(res_distance2_local[loc_idx]);// не забудьте извлечь корень
        }
    }
}
