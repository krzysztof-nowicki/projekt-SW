import numpy as np
import cv2 as cv

str = 'img_018.jpg'
path = 'train/' + str
path_1 = 'shape1.png'
path_2 = 'shape2.png'
path_3 = 'shape3.png'
path_4 = 'shape4.png'
path_5 = 'shape5.png'
shapeimg1 = cv.imread(path_1, 0)
testimg1 = cv.imread(path_1)
shapeimg1 = cv.GaussianBlur(shapeimg1, (3, 3), 0)
shapeimg1 = cv.inRange(shapeimg1, 0, 100)
shapeimg2 = cv.imread(path_2, 0)
testimg2 = cv.imread(path_2)
shapeimg2 = cv.GaussianBlur(shapeimg2, (3, 3), 0)
shapeimg2 = cv.inRange(shapeimg2, 0, 100)
shapeimg3 = cv.imread(path_3, 0)
testimg3 = cv.imread(path_3)
shapeimg3 = cv.GaussianBlur(shapeimg3, (3, 3), 0)
shapeimg3 = cv.inRange(shapeimg3, 0, 100)
shapeimg4 = cv.imread(path_4, 0)
testimg4 = cv.imread(path_4)
shapeimg4 = cv.GaussianBlur(shapeimg4, (3, 3), 0)
shapeimg4 = cv.inRange(shapeimg4, 0, 100)
shapeimg5 = cv.imread(path_5, 0)
testimg5 = cv.imread(path_5)
shapeimg5 = cv.GaussianBlur(shapeimg5, (3, 3), 0)
shapeimg5 = cv.inRange(shapeimg5, 0, 100)
shape1, h1 = cv.findContours(shapeimg1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
shape2, h2 = cv.findContours(shapeimg2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
shape3, h3 = cv.findContours(shapeimg3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
shape4, h4 = cv.findContours(shapeimg4, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
shape5, h5 = cv.findContours(shapeimg5, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(testimg1, shape1, -1, (0, 255, 0), -1)
cv.drawContours(testimg2, shape2, -1, (0, 255, 0), -1)
cv.drawContours(testimg3, shape3, -1, (0, 255, 0), -1)
cv.drawContours(testimg4, shape4, -1, (0, 255, 0), -1)
cv.drawContours(testimg5, shape5, -1, (0, 255, 0), -1)

img = cv.imread(path)
img = cv.resize(img, None, fx=0.25, fy=0.25)

img = cv.pyrMeanShiftFiltering(img, 10, 21)

img = img[:, :, 0]

img = cv.inRange(img, 0, 100)
img2 = cv.imread(path)
img2 = cv.resize(img2, None, fx=0.25, fy=0.25)
img2 = cv.pyrMeanShiftFiltering(img2, 10, 21)

img2 = img2[:, :, 2]
img2 = cv.inRange(img2, 0, 100)

img3 = cv.imread(path)
img3 = cv.resize(img3, None, fx=0.25, fy=0.25)
img3 = cv.pyrMeanShiftFiltering(img3, 10, 21)
img3 = img3[:, :, 2]
img3 = 255 - img3
img3 = cv.inRange(img3, 0, 80)
kernel = np.ones((3, 3), np.uint8)
img4 = cv.bitwise_or(img, img2)


img4 = cv.morphologyEx(img4, cv.MORPH_CLOSE, kernel)
draw = cv.imread(path)
draw = cv.resize(draw, None, fx=0.25, fy=0.25)

contours, hierarchy = cv.findContours(img4, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

shape1_count = 0
shape1_cont = []
shape2_count = 0
shape2_cont = []
shape3_count = 0
shape3_cont = []
shape4_count = 0
shape4_cont = []
shape5_count = 0
shape5_cont = []
yellow_cnt = 0
red_cnt = 0
green_cnt = 0
blue_cnt = 0
white_cnt = 0
mixed_cnt = 0
for contour in contours:
    area = cv.contourArea(contour)
    if area > 1500.0:
        color_que=0
        yellow = 0
        red = 0
        green = 0
        blue = 0
        white = 0
        mixed = 0
        x, y, w, h = cv.boundingRect(contour)
        bboxhsv = draw[y:y + h, x:x + w]
        bboxhsv = cv.cvtColor(bboxhsv, cv.COLOR_BGR2HSV)

        h, s, v = bboxhsv[:, :, 0], bboxhsv[:, :, 1], bboxhsv[:, :, 2]

        hist_h = cv.calcHist([h], [0], None, [179], [0, 179])

        mat_hard = cv.matchShapes(contour, shape1[0], 1, 0.0)

        mat = cv.matchShapes(contour, shape2[0], 1, 0.0)

        if mat < 0.15:
            shape2_cont.append(contour)
            shape2_count += 1
            color_que += 1
        mat = cv.matchShapes(contour, shape5[0], 1, 0.0)

        if mat < 0.068:
            shape5_cont.append(contour)
            shape5_count += 1
            color_que+=1
        mat = cv.matchShapes(contour, shape3[0], 1, 0.0)

        if mat < 0.2:
            shape3_cont.append(contour)
            shape3_count += 1
            color_que += 1
        elif mat_hard < 0.4:
            shape1_cont.append(contour)
            shape1_count += 1
            color_que += 1
        mat = cv.matchShapes(contour, shape4[0], 1, 0.0)

        if mat < 0.15:
            shape4_cont.append(contour)
            shape4_count += 1
            color_que += 1

        if color_que > 0:
            for i in range(len(hist_h)):
                if hist_h[i] > 400:
                    if 17 < i < 27:
                        yellow += 1
                        if yellow <= 1:
                            mixed += 1
                    elif i < 17 or i > 156:
                        red += 1
                        if red <= 1:
                            mixed += 1
                    elif 37 < i < 79:
                        green += 1
                        if green <= 1:
                            mixed += 1
                    elif 85 < i < 126:
                        blue += 1
                        if blue <= 1:
                            mixed += 1
                    else:
                        white += 1
                        if white <= 1:
                            mixed += 1

            if mixed >= 2:
                mixed_cnt += 1
                print("zmieszany")
            elif yellow >= 1:
                yellow_cnt += 1
                print("zolty")

            elif red >= 1:
                red_cnt += 1
                print("red")
            elif green >= 1:
                green_cnt += 1
                print("gren")
            elif blue >= 1:
                blue_cnt += 1
                print("blu")
            elif white >= 1:
                white_cnt += 1
                print("zolty")

cv.drawContours(draw, shape1_cont, -1, (0, 255, 0), -1)
cv.drawContours(draw, shape2_cont, -1, (0, 0, 255), -1)
cv.drawContours(draw, shape3_cont, -1, (255, 0, 0), -1)
cv.drawContours(draw, shape4_cont, -1, (0, 125, 105), -1)
cv.drawContours(draw, shape5_cont, -1, (165, 0, 155), -1)
cv.imshow('window', draw)
cv.waitKey()
