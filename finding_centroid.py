import numpy as np
import cv2
import sys
from decimal import Decimal

device = cv2.VideoCapture(0)
while True:

    _, frame = device.read()
    blurorange_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    thresh = cv2.threshold(blurorange_frame, 1, 255, cv2.THRESH_BINARY)[1]
    hsv = cv2.cvtColor(blurorange_frame, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([0, 203, 147])  # (0, 203, 147) is for orange
    upper_orange = np.array([186, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    res = cv2.bitwise_and(blurorange_frame, frame, mask=mask)

    for contour in contours:

        area = cv2.contourArea(contour)
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if area > 500:

            cv2.drawContours(blurorange_frame, contour, -1, (0, 255, 0), 3)
            cv2.drawContours(res, [approx], -1, (255, 0, 0), 3)

    M = cv2.moments(mask)

    if M["m00"] != 0:

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    else:

        cX = int(M["m10"])
        cY = int(M["m01"])

    cv2.circle(mask, (cX, cY), 5, (0, 255, 255), -1)

    cv2.imshow("frame", blurorange_frame)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

device.release()
cv2.destroyAllWindows()
