---
title: 表格旋转&检测demo
date: 2020-02-10 10:09:47
tags:
---

# 表格检测部分

## 图片预处理的Python代码，接着旋转模型旋转校正后的图片，需要转成java

{% codeblock lang:python %}
# 读取预测图片
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

h, w = img.shape[:2]
size = 800.0
scale = size / min(h, w)
if h < w:
    newh, neww = size, scale * w
else:
    newh, neww = scale * h, size
if max(newh, neww) > 1333:
    scale = 1333 * 1.0 / max(newh, neww)
    newh = newh * scale
    neww = neww * scale
neww = int(neww + 0.5)
newh = int(newh + 0.5)

resized_img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_LINEAR)

if img.ndim == 3 and resized_img.ndim == 2:
    resized_img = resized_img[:, :, np.newaxis]
{% endcodeblock %}

## pb模型调用
输入的 单张 图片的维度信息：[1, 3, ?, ?], 问号表示高度和宽度
输出结点名称： 'output/boxes', 'output/scores', 'output/labels', 'output/masks'
输出结点维度： [?, ], [?, ], [?, 4], [?, 28, 28]，问号表示一张图片里面表格对象的个数

## 图像后处理部分，根据预测结果计算出四个顶点
{% codeblock lang:python %}

import tensorflow as tf
import cv2
import os
import time
import numpy as np
from common import CustomResize
from scipy import interpolate
from tensorflow.python.framework import graph_util
from after_proc import img_proc

def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box

def _paste_mask(box, mask, shape):
    """
    Args:
        box: 4 float
        mask: MxM floats
        shape: h,w
    Returns:
        A uint8 binary image of hxw.
    """
    assert mask.shape[0] == mask.shape[1], mask.shape

    ACCURATE_PASTE = True

    if ACCURATE_PASTE:
        # This method is accurate but much slower.
        mask = np.pad(mask, [(1, 1), (1, 1)], mode='constant')
        box = _scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

        mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        h, w = shape
        ys = np.arange(0.0, h) + 0.5
        xs = np.arange(0.0, w) + 0.5
        ys = (ys - box[1]) / (box[3] - box[1]) * mask.shape[0]
        xs = (xs - box[0]) / (box[2] - box[0]) * mask.shape[1]
        # Waste a lot of compute since most indices are out-of-border
        res = mask_continuous(xs, ys)
        return (res >= 0.5).astype('uint8')

def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)

result = sess.run(output_tensors, feed_dict={"image:0": resized_img})

scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])
boxes = result[0] / scale
boxes = clip_boxes(boxes, img.shape[:2])
result[0] = boxes

masks = result[3]
full_masks = [_paste_mask(box, mask, img.shape[:2])
              for box, mask in zip(boxes, masks)]
masks = full_masks
result[3] = masks

{% endcodeblock %}

## opencv计算四个顶点
{% codeblock lang:python %}
def GeneralEquation(first_x, first_y, second_x, second_y):
    # 一般式 Ax+By+C=0
    A = second_y-first_y
    B = first_x-second_x
    C = second_x * first_y - first_x * second_y
    return A, B, C


def GetIntersectPointofLines(x1, y1, x2, y2, x3, y3, x4, y4):
    A1, B1, C1 = GeneralEquation(x1, y1, x2, y2)

    A2, B2, C2 = GeneralEquation(x3, y3, x4, y4)

    m = A1 * B2 - A2 * B1

    if m == 0:
        print("直线平行，无交点")
        return None, None
    else:
        x = C2 / m * B1 - C1 / m * B2
        y = C1 / m * A2 - C2 / m * A1
        return x, y


def dist(key_point):
    left = key_point[0]
    right = key_point[1]
    return math.sqrt(math.pow((left[0] - right[0]), 2) + math.pow((left[1] - right[1]), 2))


def cpt_line(points):
    lines = []
    pt_num = len(points)
    for p in range(pt_num):
        q = (p + 1) % pt_num
        left = points[p]
        right = points[q]
        lines.append([left, right, p])
    return lines


def is_vertical(line):
    left = line[0]
    right = line[1]
    deta_x = abs(right[0] - left[0])
    deta_y = abs(right[1] - left[1])
    if deta_y > deta_x:
        return True
    else:
        return False


def new_order(points):
    new_points = []
    avg_x = np.mean(points[:, 0])
    avg_y = np.mean(points[:, 1])
    lines = cpt_line(points)
    lines = sorted(lines, key=lambda k: dist(k), reverse=False)

    # 倾斜角度小于45度下成立，最短边在左右两边，使用竖直分割线
    if is_vertical(lines[0]):
        left_points = []
        right_points = []
        for point in points:
            if point[0] < avg_x:
                left_points.append(point)
            else:
                right_points.append(point)

        assert len(left_points) == 2, "顶点排序出错，请检查new_order函数"

        left_points = sorted(left_points, key=lambda k: k[1], reverse=False)
        right_points = sorted(right_points, key=lambda k: k[1], reverse=False)
        new_points.append(left_points[0])
        new_points.append(right_points[0])
        new_points.append(right_points[1])
        new_points.append(left_points[1])
    # 倾斜角度小于45度下成立，最短边在上下两边，使用水平分割线
    else:
        up_points = []
        down_points = []
        for point in points:
            if point[1] < avg_y:
                up_points.append(point)
            else:
                down_points.append(point)

        assert len(up_points) == 2, "顶点排序出错，请检查new_order函数"

        up_points = sorted(up_points, key=lambda k: k[0], reverse=False)
        down_points = sorted(down_points, key=lambda k: k[0], reverse=False)
        new_points.append(up_points[0])
        new_points.append(up_points[1])
        new_points.append(down_points[1])
        new_points.append(down_points[0])
    return np.array(new_points)


def img_proc(image, thresh, file, index):

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)

    contours = sorted(contours, key=lambda k: len(k), reverse=True)

    num = 1
    for i in range(num):
        cnt = contours[i]

        acc = 30
        for j in range(5):
            approx = cv2.approxPolyDP(cnt, acc, True)
            acc += 50
            if len(approx) <= 8:
                break
        approx = np.squeeze(approx, axis=1)

        #image = cv2.polylines(image, np.int32([approx]), True, (0, 0, 255), 6)

        assert len(approx) >= 4, "轮廓提取出错，少于4个顶点"

        if len(approx) != 4:
            lines = cpt_line(approx)

            vet_lines = []
            hor_lines = []
            for line in lines:
                if is_vertical(line):
                    vet_lines.append(line)
                else:
                    hor_lines.append(line)

            vet_lines = sorted(vet_lines, key=lambda k: dist(k), reverse=True)
            hor_lines = sorted(hor_lines, key=lambda k: dist(k), reverse=True)

            assert len(vet_lines) >= 2, '线条分类错误'
            assert len(hor_lines) >= 2, '线条分类错误'

            lines = []
            lines.append(vet_lines[0])
            lines.append(hor_lines[0])
            lines.append(vet_lines[1])
            lines.append(hor_lines[1])
            #lines = sorted(lines[:4], key=lambda k: k[2], reverse=False)

            ans_points = []
            for p in range(4):
                q = (p + 1) % 4
                line1 = lines[p]
                line2 = lines[q]
                x1, y1, x2, y2 = line1[0][0], line1[0][1], line1[1][0], line1[1][1]
                x3, y3, x4, y4 = line2[0][0], line2[0][1], line2[1][0], line2[1][1]
                x, y = GetIntersectPointofLines(x1, y1, x2, y2, x3, y3, x4, y4)
                ans_points.append([x, y])
            ans_points = np.array(ans_points)
        else:
            ans_points = approx

        assert len(ans_points) == 4, "提取顶点出错，必须是4个顶点，请检查直线交点计算过程"

        #print('ans_points: ', ans_points)
        #image = cv2.polylines(image, np.int32([ans_points]), True, (0, 255, 0), 6)

        ans_points = new_order(ans_points)
        #print('new_ans_points: ', ans_points)

        #ans_points = shrink_poly(ans_points)
        #image = cv2.polylines(image, np.int32([ans_points]), True, (0, 0, 255), 5)

        return ans_points

for i, mask in enumerate(masks):
    thresh = np.where(mask > 0, 255, 0).astype('uint8')
    ans_points = img_proc(img, thresh, file, i)
{% endcodeblock %}
