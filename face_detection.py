import cv2 as cv

face_haar_cascade = cv.CascadeClassifier('face.xml')

cap = cv.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
