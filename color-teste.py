import cv2 as cv
import numpy as np

# Rangos LAB para cada color (ajustables según cámara/iluminación)
black_low = np.array([0, 0, 0], np.uint8)
black_high = np.array([50, 255, 255], np.uint8)
blue_low = np.array([0, 100, 0], np.uint8)
blue_high = np.array([255, 150, 100], np.uint8)
green_low = np.array([0, 50, 100], np.uint8)
green_high = np.array([255, 100, 150], np.uint8)
red_low = np.array([0, 150, 150], np.uint8)
red_high = np.array([255, 200, 200], np.uint8)
purple_low = np.array([0, 150, 50], np.uint8)
purple_high = np.array([255, 200, 150], np.uint8)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)

    mask_black = cv.inRange(lab, black_low, black_high)
    mask_blue = cv.inRange(lab, blue_low, blue_high)
    mask_green = cv.inRange(lab, green_low, green_high)
    mask_red = cv.inRange(lab, red_low, red_high)
    mask_purple = cv.inRange(lab, purple_low, purple_high)

    cv.imshow('Negro', mask_black)
    cv.imshow('Azul', mask_blue)
    cv.imshow('Verde', mask_green)
    cv.imshow('Rojo', mask_red)
    cv.imshow('Morado', mask_purple)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
