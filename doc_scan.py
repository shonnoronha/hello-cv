import cv2 as cv
import numpy as np


image_width = 480
image_height = 640


def sort_points(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), dtype=np.int32)
    _sum = points.sum(1)
    _diff = np.diff(points, axis=1)
    points_new[0] = points[np.argmin(_sum)]
    points_new[3] = points[np.argmax(_sum)]
    points_new[1] = points[np.argmin(_diff)]
    points_new[2] = points[np.argmax(_diff)]
    return points_new


def apply_warp(image, points):
    points = sort_points(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    output = cv.warpPerspective(image, matrix, (image_width, image_height))
    crop = output[20:output.shape[0]-20, 20:output.shape[1]-20]
    crop = cv.resize(crop, (image_width, image_height))
    return crop


def get_contours(image):
    biggest_contour = np.array([])
    max_area = 0
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 5000:
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.02*perimeter, True)
            if area > max_area and len(approx) == 4:
                biggest_contour = approx
                max_area = area
    cv.drawContours(img_contour, biggest_contour, -1, (0, 255, 0), 20)
    return biggest_contour


def process(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)
    canny = cv.Canny(blur, 200, 200)
    kernel = np.ones((5, 5), dtype='uint8')
    dilated = cv.dilate(canny, kernel, iterations=2)
    eroded = cv.erode(dilated, kernel, iterations=1)
    return eroded


cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    cv.resize(img, (image_width, image_height))
    img_contour = img.copy()
    mod_img = process(img)
    largest_contour = get_contours(mod_img)
    warped_image = np.zeros_like(img)
    if len(largest_contour) > 0:
        warped_image = apply_warp(img, largest_contour)
        cv.imshow('result', warped_image)
    cv.imshow('flow', img_contour)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
