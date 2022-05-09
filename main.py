import numpy as np
import cv2 as cv

str = 'img_003.jpg'
path = 'train/' + str

img = cv.imread(path)
img = cv.resize(img,None, fx=0.25, fy=0.25)
# img = cv.pyrMeanShiftFiltering(img, 10, 21)
img = cv.GaussianBlur(img, (5, 5), 0)
img = img[:,:,0]

img2 = cv.imread(path)
img2 = cv.resize(img2,None, fx=0.25, fy=0.25)
# img2 = cv.pyrMeanShiftFiltering(img2, 10, 21)
img2 = cv.GaussianBlur(img2, (5, 5), 0)
img2 = img2[:,:,2]

# img2 = 255-img
img3 = cv.imread(path)
img3 = cv.resize(img3,None, fx=0.25, fy=0.25)
# img3 = cv.pyrMeanShiftFiltering(img3, 10, 21)
img3 = cv.GaussianBlur(img3, (5, 5), 0)
img3 = img3[:,:,0]

img3 = 255-img3
draw = cv.imread(path)
draw = cv.resize(draw,None, fx=0.25, fy=0.25)

#normal 90 - 200
#inv 87 - 170



kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(img,kernel,iterations = 1)
ret, img = cv.threshold(img, 90, 200, cv.THRESH_BINARY)
img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

ret2, img2 = cv.threshold(img2, 100, 200, cv.THRESH_BINARY)
img2 = cv.morphologyEx(img2, cv.MORPH_CLOSE, kernel)

ret3, img3 = cv.threshold(img3, 87, 200, cv.THRESH_BINARY)
img3 = cv.morphologyEx(img3, cv.MORPH_CLOSE, kernel)
# cv.pyrMeanShiftFiltering()
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)



# img = img/2.0
# img2 = img2/2.0
# img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# print(img)
# print(img2)
#     print(i)
#     print(j)
# print(img.shape)
# cv.imshow('window1', img)
# cv.imshow('window2', img2)
# cv.imshow('window3', img3)
# cv.waitKey()
dst = cv.addWeighted(img,0.5,img2,0.5,0)
# cv.imshow('windows2', dst)
dst = cv.addWeighted(dst,1.0,img3,0.5,0)
# print( img)
i,j = dst.shape
for k in range(i):
    for l in range(j):
        if dst[k, l] < 254:
            dst[k, l] = 0
        else:
            dst[k, l] = 255
# print(dst)
# cv.imshow('window', dst)
# cv.waitKey()

contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(contours)
cv.drawContours(draw, contours, -1, (0,255,0), 3)
cv.imshow('window', draw)
cv.waitKey()

