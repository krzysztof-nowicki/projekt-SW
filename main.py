import numpy as np
import cv2 as cv

str = 'img_015.jpg'
path = 'train/' + str

#comm

img = cv.imread(path)
img = cv.resize(img,None, fx=0.25, fy=0.25)


# print(img)
img = cv.pyrMeanShiftFiltering(img, 10, 21)
# cv.imshow('window13', img)
# img = cv.GaussianBlur(img, (3,3), 0)
# img = cv.medianBlur(img, 5)
img = img[:,:,0]

img = cv.inRange(img, 0, 100)

img2 = cv.imread(path)
img2 = cv.resize(img2,None, fx=0.25, fy=0.25)
img2 = cv.pyrMeanShiftFiltering(img2, 10, 21)
# img2 = cv.GaussianBlur(img2, (3,3), 0)
# img2 = cv.medianBlur(img2, 5)
img2 = img2[:,:,2]

# img2 = 255-img
img2 = cv.inRange(img2, 0, 100)

img3 = cv.imread(path)
img3 = cv.resize(img3,None, fx=0.25, fy=0.25)
img3 = cv.pyrMeanShiftFiltering(img3, 10, 21)
# img3 = cv.GaussianBlur(img3, (3,3), 0)
# img3 = cv.medianBlur(img3, 5)
# lower = np.array([140, 140, 140])
# upper = np.array([160, 160, 160])
# print(dst)
# dst = cv.inRange(dst, 0, 250)
img3 = img3[:,:,0]

img3 = 255-img3
# cv.imshow('window15', img3)
img3 = cv.inRange(img3, 0, 95)
# cv.imshow('window11', img2)
img4 = cv.bitwise_or(img, img2)
img4 = cv.bitwise_or(img4, img3)
# cv.imshow('window12', img4)
# cv.waitKey()
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

ret3, img3 = cv.threshold(img3, 85, 200, cv.THRESH_BINARY)
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
# dst = cv.addWeighted(img,0.5,img2,0.5,0)
# # cv.imshow('windows2', dst)
# dst = cv.addWeighted(dst,1.0,img3,0.5,0)
# # print( img)
# i,j = dst.shape
# # lower = np.array([140, 140, 140])
# # upper = np.array([160, 160, 160])
# print(dst)
# dst = cv.inRange(dst, 0, 250)
# for k in range(i):
#     for l in range(j):
#         if dst[k, l] < 254:
#             dst[k, l] = 0
#         else:
#             dst[k, l] = 255
# print(dst)
# cv.imshow('window', dst)
# cv.waitKey()

contours, hierarchy = cv.findContours(img4, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))
cv.drawContours(draw, contours, -1, (0,255,0), 3)
cv.imshow('window', draw)
cv.waitKey()

