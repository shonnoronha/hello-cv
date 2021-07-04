import cv2 as cv
import numpy as np

def add_canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 1)
    canny = cv.Canny(blur, 50, 150)
    return canny

def find_roi(image):
    height = image.shape[0]
    triangle = np.array([(200, height), (1100, height), (550, 250)])
    mask = np.zeros_like(image)
    cv.fillPoly(mask, [triangle], (255, 255, 255))
    maksed_image = cv.bitwise_and(image, mask)
    return maksed_image

def draw_lines(image, all_lines):
    for line in all_lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    return image

def make_lines_average(image, irregular_lines):
    left = []
    right = []
    for line in irregular_lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
    left_average = np.average(left, axis=0)
    right_average = np.average(right, axis=0)
    left_points = return_points_from_parametres(image, left_average)
    right_points = return_points_from_parametres(image, right_average)
    return np.array([left_points, right_points])

def return_points_from_parametres(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

cap = cv.VideoCapture('test2.mp4')
while (cap.isOpened()):
    _, img = cap.read()
    canny_img = add_canny(img)
    roi_image = find_roi(canny_img)
    lines = cv.HoughLinesP(roi_image, 2, np.pi/180, 100,np.array([]), minLineLength=40, maxLineGap=5)
    average_lines = make_lines_average(img, lines)
    draw_lines(img, average_lines)
    cv.imshow('image', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break