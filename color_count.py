import cv2 as cv
import numpy as np

# HSV color ranges for detection
azulAlto = np.array([148, 164, 121], dtype=np.uint8)
azulBajo = np.array([54, 124, 25], dtype=np.uint8)
verdeBajo = np.array([35, 80, 40], dtype=np.uint8)
verdeAlto = np.array([85, 255, 255], dtype=np.uint8)
rojoBajo = np.array([0, 100, 40], dtype=np.uint8)
rojoAlto = np.array([10, 255, 255], dtype=np.uint8)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # Create masks for each color
    mask_azul = cv.inRange(hsv, azulBajo, azulAlto)
    mask_verde = cv.inRange(hsv, verdeBajo, verdeAlto)
    mask_rojo = cv.inRange(hsv, rojoBajo, rojoAlto)

    # Morphological operations to clean up the masks
    kernel = np.ones((5, 5), np.uint8)
    mask_azul = cv.erode(mask_azul, kernel, iterations=1)
    mask_azul = cv.dilate(mask_azul, kernel, iterations=1)
    mask_verde = cv.erode(mask_verde, kernel, iterations=1)
    mask_verde = cv.dilate(mask_verde, kernel, iterations=1)
    mask_rojo = cv.erode(mask_rojo, kernel, iterations=1)
    mask_rojo = cv.dilate(mask_rojo, kernel, iterations=1)

    # Find contours for each color
    contornos_azul, _ = cv.findContours(mask_azul, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contornos_verde, _ = cv.findContours(mask_verde, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contornos_rojo, _ = cv.findContours(mask_rojo, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Counters for each color (only objects currently visible on screen)
    count_azul = 0
    count_verde = 0
    count_rojo = 0

    # Draw contours and count blue objects
    for contour in contornos_azul:
        area = cv.contourArea(contour)
        if area > 500:
            count_azul += 1
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                cv.putText(frame, "Azul", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            newcontour = cv.convexHull(contour)
            cv.drawContours(frame, [newcontour], -1, (255, 0, 0), 2)

    # Draw contours and count green objects
    for contour in contornos_verde:
        area = cv.contourArea(contour)
        if area > 500:
            count_verde += 1
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx, cy), 7, (0, 255, 0), -1)
                cv.putText(frame, "Verde", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            newcontour = cv.convexHull(contour)
            cv.drawContours(frame, [newcontour], -1, (0, 255, 0), 2)

    # Draw contours and count red objects
    for contour in contornos_rojo:
        area = cv.contourArea(contour)
        if area > 500:
            count_rojo += 1
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                cv.putText(frame, "Rojo", (cx - 20, cy - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            newcontour = cv.convexHull(contour)
            cv.drawContours(frame, [newcontour], -1, (0, 0, 255), 2)

    # Show the counters on the frame
    cv.putText(frame, f"Red objects: {count_rojo}", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv.putText(frame, f"Green objects: {count_verde}", (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(frame, f"Blue objects: {count_azul}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Combine all masks for display
    mask_total = cv.bitwise_or(mask_azul, mask_verde)
    mask_total = cv.bitwise_or(mask_total, mask_rojo)

    # Show the original image and the mask
    cv.imshow('Frame', frame)
    cv.imshow('Mask', mask_total)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()