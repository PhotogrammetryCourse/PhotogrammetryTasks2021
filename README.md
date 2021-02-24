В этом репозитории предложены задания для курса по фотограмметрии в [CSCenter](https://compscicenter.ru/courses/photogrammetry/) и [CSClub](https://compsciclub.ru/courses/photogrammetry/).

[Остальные задания](https://github.com/PhotogrammetryCourse/PhotogrammetryTasks2021/).

# Задание 2. Сопоставление ключевых точек и подсчет гомографии

[![Build Status](https://travis-ci.com/PhotogrammetryCourse/PhotogrammetryTasks2021.svg?branch=task02)](https://travis-ci.com/PhotogrammetryCourse/PhotogrammetryTasks2021)

0. Скачать себе ветку task02 и смерджить в нее свою реализацию SIFT
1. [Пересобрать OpenCV 4.5.1](https://github.com/PhotogrammetryCourse/PhotogrammetryTasks2021/blob/task02/CMakeLists.txt#L19-L31) (добавился новый модуль calib3d)
2. Выполнить задания ниже (не используйте пожалуйста C++ из будущего о котором не знает GCC 5.5 - именно он будет использоваться при тестировании в Travis CI, ориентируйтесь на C++11)
3. Отправить **Pull-request** с названием```Task02 <Имя> <Фамилия> <Аффиляция>```:

 - Скопируйте в описание [шаблон](https://raw.githubusercontent.com/PhotogrammetryCourse/PhotogrammetryTasks2021/task02/.github/pull_request_template.md)
 - Обязательно отправляйте PR из вашей ветки **task02** (вашего форка) в ветку **task02** (основного репозитория)
 - Перечислите свои мысли по вопросам поднятым в коде и просто появившиеся в процессе выполнения задания (выписывайте их с самого начала в отдельный текстовый файл, в шаблоне предложены некоторые вопросы)
 - Создайте PR
 - Затем дождавшись отработку Travis CI (около 15 минут) - скопируйте в описание PR вывод исполнения вашей программы **на CI** (через редактирование описания PR)

**Дедлайн**: 23:59 24 февраля, но очень желательно к началу лекции 23 февраля, чтобы понимать разбор задания и не отнимать времени на подготовку следующей домашки

Если времени не хватит - отправьте то, что вы успели сделать
(и мне очень поможет, если вы сможете детализировать, на чем застряли и на что ушло слишком много времени).

Задание 2.0.
=========

Ознакомьтесь со структурой проекта:

1. ```src/phg/matching/```, ```src/phg/sfm/``` - основная часть где вы будете реализовывать алгоритм

2. ```tests/test_matching.cpp``` - тесты которые будут прогонять ваш алгоритм на каких-то относительно простых манипуляциях с маленькими картинками, если вам хочется добавить другие сценарии тестирования (возможно с другими метриками) - здорово!

3. ```data/src``` - исходные данные используемые при тестировании (к ним используются относительные пути, поэтому нужно выставить Working directory = путь к проекту)

4. ```data/debug/test_matching``` - сюда тесты сохранят картинки с визуализацией результата

5. ```data/debug/test_matching``` - сюда вам предлагается сохранять любые промежуточные картинки-визуализации, это очень полезно для отладки, оценки качества, уверенности и в целом один из немногих способов качественно "заглянуть в черную коробку"

Задание 2.1.
=========

1. Убедитесь что у вас все компилируется и тесты проходят.

2. Ознакомьтесь с тем как проводится тестирование - ```tests/test_matching.cpp```:

3. Обратите внимание что там в качестве метода фильтрации метчей, полученных поиском ближайших соседей, используется [GMS matcher](https://github.com/JiawangBian/GMS-Feature-Matcher). Это метод, идейно похожий на Cluster filtering, который вам предстоит реализовать, но построен немного иначе и не ищет явно в каждой точке ближайших соседей. Потенциально он может лучше работать в случае, когда очень много шумных матчей - тогда даже если хорошие сопоставления расположены плотно, между ними втискиваются шумные матчи и уменьшают размер пересечения множеств ближайших соседей. Вы можете попробовать проверить, так ли это, реализовав свой Cluster filtering, включив ENABLE_MY_MATCHING и найдя тест, в котором переменная good_clusters обращается в ноль. (тест при этом может проходить, так как для оценки гомографии включены одновременно и Ratio-test и Cluster filtering, за это отвечает переменная good_ratio_and_clusters)

4. Посмотрите на картинки, которые сохранились в папку ```data/debug/test_matching```. В частности, интересно посмотреть, насколько большая разница между шумными матчами, полученными методом поиска ближайших соседей и прошедшими фильтрацию. В вашем задании нужно будет добиться сопоставимого качества с помощью Ratio-test и Cluster filtering.


Задание 2.2.
=========

1. Включите тестирование вашего матчинга - см. **ENABLE_MY_MATCHING** в ```test/test_matching.cpp```
2. Реализуйте пропущенные участки алгоритма, проверьте, что все тесты проходят
3. Включите тестирование вашего SIFT - см. **ENABLE_MY_DESCRIPTOR** в ```test/test_matching.cpp```
4. Проверьте, что тесты продолжают проходить. Если это не так, постарайтесь понять в чем причина с помощью отладочного вывода (просмотра дебаговых картинок, сравнения количества найденных сопоставлений на разных шагах матчинга с разными дескрипторами и тд.). Если заставить работать не получилось, коммитьте решение с выключенным **ENABLE_MY_DESCRIPTOR** чтобы проходили тесты. Баллы в этом случае не снимаются, но у вас может остаться неприятное чувство неполноценности проекта как самостоятельной единицы.

 - Если все хорошо, за выполненное задание дается **8 баллов**
 - **1 доп. балл** можно получить, если при оценке матрицы гомографии реализовать метод **A contrario RANSAC**, не требующий на вход порога (см. homography.cpp:166)
 - **1 доп. балл** можно получить, если реализовать Brute-force матчер на GPU. Для включения его в тестах см. **ENABLE_GPU_BRUTEFORCE_MATCHER** в ```test/test_matching.cpp```
