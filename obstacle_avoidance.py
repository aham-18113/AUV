from collections import deque
from imutils.video import VideoStream
import numpy as np
from argparse import ArgumentParser
import cv2
import imutils
import time

greenLower = (38, 102, 84)
greenUpper = (74, 244, 147)

blueLower = (38, 102, 84)
blueUpper = (74, 244, 147)
pts = deque(maxlen=64)
bpts = deque(maxlen=64)


vs1 = VideoStream(src=0).start()
vs2 = VideoStream(src=0).start()


def shape_detect(frame):
    shape = "unidentified"
    resized = imutils.resize(frame, width=300)
    ratio = frame.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) >= 5:
            shape = "circle"
        try:
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
            cv2.putText(
                frame,
                shape,
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )
            return frame
        except ZeroDivisionError as e:
            pass
    return frame


def detect_color(frame, lower, upper, deq):
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    (h, w) = frame.shape[:2]
    cv2.circle(frame, (w // 2, h // 2), 3, (255, 255, 255), -1)
    (h, w) = frame.shape[:2]
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
    deq.appendleft(center)
    return deq


def grid(frame, deq):
    (h, w) = frame.shape[:2]  # h=y-axis, w=x-axis
    if deq[i][0] < (w / 3) and deq[i][1] < (h / 3):
        print("up left")

    if deq[i][0] > (w / 3) and deq[i][0] < (2 * w / 3) and deq[i][1] < (h / 3):
        print("up")

    if deq[i][0] > (2 * w / 3) and deq[i][1] < (h / 3):
        print("up right")

    if deq[i][0] < (w / 3) and deq[i][1] > (h / 3) and deq[i][1] < (2 * h / 3):
        print("left")

    if (
        deq[i][0] > (w / 3)
        and deq[i][0] < (2 * w / 3)
        and deq[i][1] > (h / 3)
        and deq[i][1] < (2 * h / 3)
    ):
        print("center")

    if deq[i][0] > (2 * w / 3) and deq[i][1] > (h / 3) and deq[i][1] < (2 * h / 3):
        print("right")

    if deq[i][0] < (w / 3) and deq[i][1] > (2 * h / 3):
        print("down left")

    if deq[i][0] > (w / 3) and deq[i][0] < (2 * w / 3) and deq[i][1] > (2 * h / 3):
        print("down")

    if deq[i][0] > (2 * w / 3) and deq[i][1] > (2 * h / 3):
        print("down right")


time.sleep(2.0)


while True:
    color = (255, 0, 0)
    thickness = 1
    frame = vs1.read()
    if frame is None:
        break
    pts = detect_color(frame, greenLower, greenUpper, pts)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        grid(frame, pts)
    while len(pts) == 0:
        frame1 = vs2.read()
        if frame1 is None:
            break
        bpts = detect_color(frame1, blueLower, blueUpper, bpts)
        for i in range(1, len(bpts)):
            if bpts[i - 1] is None or bpts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
            cv2.line(frame1, bpts[i - 1], bpts[i], (0, 0, 255), thickness)
            counter = 0
            grid(frame1, bpts)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if not ArgumentParser("video", False):
    vs.stop()
    vs1.stop()
else:
    vs.release()
    vs1.release()
cv2.destroyAllWindows()
