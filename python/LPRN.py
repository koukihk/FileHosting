import cv2
import numpy as np
import pytesseract
import os
import time


# 读取图片
def imread_photo(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, flags)


# 等比缩放
def resize_keep_aspectratio(image_src, dst_size):
    src_h, src_w = image_src.shape[:2]
    dst_h, dst_w = dst_size
    h = dst_w * (float(src_h) / src_w)
    w = dst_h * (float(src_w) / src_h)
    h = int(h)
    w = int(w)
    if h <= dst_h:
        image_dst = cv2.resize(image_src, (dst_w, int(h)))
    else:
        image_dst = cv2.resize(image_src, (int(w), dst_h))
    h_, w_ = image_dst.shape[:2]
    print('等比缩放完毕')
    return image_dst


# 调整尺寸
def resize_photo(imgArr, MAX_WIDTH=1000):
    img = imgArr
    rows, cols = img.shape[:2]  # 获取输入图像的高和宽
    # 如果宽度大于设定的阈值
    if cols > MAX_WIDTH:
        change_rate = MAX_WIDTH / cols
        img = cv2.resize(img, (MAX_WIDTH, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    return img


# HSV
def hsv_color_find(img):
    img_copy = img.copy()
    hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([100, 80, 80])
    high_hsv = np.array([124, 255, 255])
    # 设置HSV的阈值
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    cv2.imshow("hsv_color_find", mask)
    # 将掩膜与图像层逐像素相加
    res = cv2.bitwise_and(img_copy, img_copy, mask=mask)
    cv2.imshow("hsv_color_find2", res)

    print('hsv提取蓝色部分完毕')

    return res


# 寻找车牌潜在区域
def predict(imageArr):
    img_copy = imageArr.copy()
    img_copy = hsv_color_find(img_copy)
    # RGB->灰度
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
    kernel = np.ones((23, 23), np.uint8)
    # 侵蚀和膨胀
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    # 计算两个数组的加权和
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)

    cv2.imshow("img_opening", img_opening)

    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, img_thresh2 = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY)

    cv2.imshow("img_thresh", img_thresh)
    cv2.imshow("img_thresh2", img_thresh2)

    # 查找边缘
    img_edge = cv2.Canny(img_thresh, 100, 200)

    # 开运算和闭运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    img_edge3 = cv2.morphologyEx(img_thresh2, cv2.MORPH_CLOSE, kernel)
    img_edge4 = cv2.morphologyEx(img_edge3, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("img_edge3", img_edge3)
    cv2.imshow("img_edge4", img_edge4)

    # 查找图像边缘整体形成的矩形区域
    contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours2, hierarchy2 = cv2.findContours(img_edge4, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print('可能是车牌的一些矩形区域提取完毕')

    return gray_img_, contours, contours2


# 画出轮廓
def draw_contours(img, contours):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 获得最小的矩形轮廓 可能带旋转角度
        rect = cv2.minAreaRect(c)
        # 计算最小区域的坐标
        box = cv2.boxPoints(rect)
        # 坐标规范化为整数
        box = np.int0(box)
        # 画出轮廓
        cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

    cv2.imshow("contours", img)


# 矩形过滤
def chose_licence_plate(contours, Min_Area=2000):
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:
            temp_contours.append(contour)
    car_plate1 = []
    car_plate2 = []
    car_plate3 = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height

        if aspect_ratio > 1.5 and aspect_ratio < 4.65:
            car_plate1.append(temp_contour)
            rect_vertices = cv2.boxPoints(rect_tupple)
            rect_vertices = np.int0(rect_vertices)
            # print(temp_contour)
    print('一次筛查后，符合比例的矩形有' + str(len(car_plate1)) + '个')

    # 二次筛查 如果符合尺寸的矩形大于1，则缩小宽高比
    if len(car_plate1) > 1:
        for temp_contour in car_plate1:
            rect_tupple = cv2.minAreaRect(temp_contour)
            rect_width, rect_height = rect_tupple[1]
            if rect_width < rect_height:
                rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height

            # 车牌正常情况下宽高比在2 - 3.15之间 稍微放宽点范围
            if aspect_ratio > 1.6 and aspect_ratio < 4.15:
                car_plate2.append(temp_contour)
                rect_vertices = cv2.boxPoints(rect_tupple)
                rect_vertices = np.int0(rect_vertices)
    print('二次筛查后，符合比例的矩形还有' + str(len(car_plate2)) + '个')

    # 三次筛查 如果符合尺寸的矩形大于1，则缩小宽高比
    if len(car_plate2) > 1:
        for temp_contour in car_plate2:
            rect_tupple = cv2.minAreaRect(temp_contour)
            rect_width, rect_height = rect_tupple[1]
            if rect_width < rect_height:
                rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height

            # 车牌正常情况下宽高比在2 - 3.15之间 稍微放宽点范围
            if aspect_ratio > 1.8 and aspect_ratio < 3.35:
                car_plate3.append(temp_contour)
                rect_vertices = cv2.boxPoints(rect_tupple)
                rect_vertices = np.int0(rect_vertices)
    print('三次筛查后，符合比例的矩形还有' + str(len(car_plate3)) + '个')

    if len(car_plate3) > 0:
        return car_plate3
    if len(car_plate2) > 0:
        return car_plate2
    return car_plate1


# 车牌截取
def license_segment(car_plates, out_path):
    i = 0
    if len(car_plates) == 1:
        for car_plate in car_plates:
            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
            cv2.rectangle(img, (row_min, col_min), (row_max, col_max), (0, 255, 0), 2)
            card_img = img[col_min:col_max, row_min:row_max, :]
            cv2.imwrite(out_path + "/card_img" + str(i) + ".jpg", card_img)
            cv2.imshow("card_img" + str(i) + ".jpg", card_img)
            i += 1
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print('共切出' + str(i) + '张车牌图。')

    return out_path + "/card_img0.jpg"


# 根据设定的阈值和图片直方图，找出波峰
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


# 将截取到的车牌照片转化为灰度图，确定上下边框
def remove_plate_upanddown_border(card_img):
    plate_Arr = cv2.imread(card_img)
    plate_gray_Arr = cv2.cvtColor(plate_Arr, cv2.COLOR_BGR2GRAY)
    ret, plate_binary_img = cv2.threshold(plate_gray_Arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
    row_min = np.min(row_histogram)
    row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
    row_threshold = (row_min + row_average) / 2
    wave_peaks = find_waves(row_threshold, row_histogram)
    # 挑选跨度最大的波峰
    wave_span = 0.0
    for wave_peak in wave_peaks:
        span = wave_peak[1] - wave_peak[0]
        if span > wave_span:
            wave_span = span
            selected_wave = wave_peak
    plate_binary_img = plate_binary_img[selected_wave[0]:selected_wave[1], :]
    cv2.imshow("plate_binary_img", plate_binary_img)

    return plate_binary_img



# 二分-K均值聚类算法

def distEclud(vecA, vecB):
    return np.sum(abs(vecA - vecB))


def randCent(dataSet, k):
    n = dataSet.shape[1]  # 列数
    centroids = np.zeros((k, n))  # 用来保存k个类的质心
    for j in range(n):
        minJ = np.min(dataSet[:, j], axis=0)
        rangeJ = float(np.max(dataSet[:, j])) - minJ
        for i in range(k):
            centroids[i:, j] = minJ + rangeJ * (i + 1) / k
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))  # 一列记录簇索引值，第二列存储误差
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroid0 = np.mean(dataSet, axis=0).tolist()
    centList = []
    centList.append(centroid0)
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.array(centroid0), dataSet[j, :]) ** 2
    while len(centList) < k:  # 小于K个簇时
        lowestSSE = np.inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0] == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = np.sum(splitClustAss[:, 1])
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:, 0] != i), 1])
            if (sseSplit + sseNotSplit) < lowestSSE:  # 如果满足，则保存本次划分
                bestCentTosplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentTosplit
        centList[bestCentTosplit] = bestNewCents[0, :].tolist()
        centList.append(bestNewCents[1, :].tolist())
        clusterAssment[np.nonzero(clusterAssment[:, 0] == bestCentTosplit)[0], :] = bestClustAss
    return centList, clusterAssment


# 字符分割
def split_licensePlate_character(plate_binary_img):
    plate_binary_Arr = np.array(plate_binary_img)
    row_list, col_list = np.nonzero(plate_binary_Arr >= 255)
    dataArr = np.column_stack((col_list, row_list))  # dataArr的第一列是列索引，第二列是行索引，要注意
    centroids, clusterAssment = biKmeans(dataArr, 7, distMeas=distEclud)
    centroids_sorted = sorted(centroids, key=lambda centroid: centroid[0])
    split_list = []
    for centroids_ in centroids_sorted:
        i = centroids.index(centroids_)
        current_class = dataArr[np.nonzero(clusterAssment[:, 0] == i)[0], :]
        x_min, y_min = np.min(current_class, axis=0)
        x_max, y_max = np.max(current_class, axis=0)
        split_list.append([y_min, y_max, x_min, x_max])
    character_list = []
    for i in range(len(split_list)):
        single_character_Arr = plate_binary_img[split_list[i][0]: split_list[i][1], split_list[i][2]:split_list[i][3]]
        character_list.append(single_character_Arr)
        cv2.imshow('character' + str(i), single_character_Arr)
        # 存储所有字符切图
        cv2.imwrite('img/LPR/character' + str(i) + '.jpg', single_character_Arr)
    print('字符切割完毕')
    return character_list  # character_list中保存着每个字符的二值图数据



# Tesseract-OCR 图像识别
def tesseract_ocr(car_img_path):
    print('\nTesseract_OCR识别完成，结果如下：')
    ret = os.popen('C:\Program Files\Tesseract-OCR\\tesseract.exe ' + car_img_path + ' result -l chi_sim')
    # 处理延时
    time.sleep(1)
    # 读写模式打开文件
    with open('result.txt', 'r', encoding='utf-8') as f:
        # 读取第一行
        line1 = f.readline()
        rows = len(f.readlines())
        # print(rows)
        if rows > 0:
            print('车牌为：' + line1 + '\n')
        else:
            print('识别失败\n')

# 配合pytesseract食用 需要配置Tesseract-OCR的环境变量
def pytesseract_ocr(car_img_path):
     print('\n函数pytesseract_ocr识别结果如下：')
     img_cv = cv2.imread(car_img_path)

     # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
     # we need to convert from BGR to RGB format/mode:
     img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
     ret = pytesseract.image_to_string(img_rgb, lang='chi_sim')
     print('车牌为：' + ret + '\n')

if __name__ == "__main__":
    # 你要识别的图片
    img = imread_photo("img/LPR/car02.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', img)
    cv2.imshow('gray_img', gray_img)

    # 调整图像的尺寸大小 等比缩放至500*500
    img = resize_keep_aspectratio(img, [500, 500])
    gray_img = resize_keep_aspectratio(gray_img, [500, 500])

    # 过一系列的处理，找到可能是车牌的一些矩形区域
    gray_img_, contours, contours2 = predict(img)
    cv2.imshow('gray_img_', gray_img_)

    # 画出轮廓
    # draw_contours(gray_img_, contours)
    draw_contours(gray_img, contours2)

    # 根据车牌的一些物理特征（面积等）对所得的矩形进行过滤
    car_plate = chose_licence_plate(contours2)

    if len(car_plate) == 0:
        print('没有识别到车牌，程序结束。')
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 根据得到的车牌定位，将车牌从原始图像中截取出来，并存在当前目录中。
        car_img_path = license_segment(car_plate, "img/LPR")

        # 将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
        plate_binary_img = remove_plate_upanddown_border(car_img_path)

        # 对车牌的二值图进行水平方向的切分，将字符分割出来
        character_list = split_licensePlate_character(plate_binary_img)

        # Tesseract-OCR 图像识别
        # pytesseract_ocr(car_img_path)

        text = '车牌字符处理完毕，请调用百度OCR进行文字识别！'
        print(text)


        cv2.waitKey(0)
        cv2.destroyAllWindows()
