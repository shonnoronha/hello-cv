import cv2 as cv
import numpy as np

# list of colors that can be detected
names = ['orange', 'aqua', 'peach', 'green']
# hsv min and max values for colors
all_colors = [[0, 110, 153, 19, 240, 255],
              [80, 151, 51, 179, 255, 255],
              [0, 0, 0, 179, 171, 71],
              [60, 116, 15, 90, 255, 255]]
# BGR values of colors
color_names = [[51, 153, 255],
               [255, 255, 0],
               [180, 229, 255],
               [51, 255, 51]]
# points to be plotted on canvas
points = []


def find_color(image, colors, c_values):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    new_points = []
    for i, cor in enumerate(colors):
        lower = np.array(cor[:3])
        upper = np.array(cor[3:])
        mask = cv.inRange(hsv, lower, upper)
        x, y = get_pos(mask)
        cv.circle(result, (x, y), 10, c_values[i], -1)
        if x != 0 and y != 0:
            new_points.append([x, y, i])
            return new_points


def get_pos(image):
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for c in contours:
        area = cv.contourArea(c)
        if area > 500:
            perimeter = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * perimeter, True)
            x, y, w, h = cv.boundingRect(approx)
    return x+(w//2), y


def draw_on_canvas(plane_points):
    for p in plane_points:
        cv.circle(result, (p[0], p[1]), 10, color_names[p[2]], -1)


cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    result = img.copy()
    received_points = find_color(img, all_colors, color_names)
    if received_points != 0 and received_points is not None:
        print(received_points)
        for point in received_points:
            points.append(point)
    if len(points) != 0:
        draw_on_canvas(points)
    cv.imshow('res', result)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
