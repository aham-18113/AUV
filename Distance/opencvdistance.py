from turtle import distance
import cv2
import numpy as np

thvalue = [1500, 1500, 1500, 1500]


def forward():
    if thvalue[0] < 1850 and thvalue[1] < 1850:
        thvalue[0] = thvalue[0] + 5
        thvalue[1] = thvalue[1] + 5
    print("forward")


def right():
    if thvalue[0] < 1850 and thvalue[1] < 1850:
        thvalue[0] = thvalue[0] + 5
        thvalue[1] = thvalue[1] + 5
    print("right")


cap = cv2.VideoCapture(0)
lower_range = np.array([0, 203, 147])  # color is orange
upper_range = np.array([186, 255, 255])


Known_distance = 13.0
Known_width = 3.0


def Focal_Length_Finder(Known_distance, real_width, width_in_rf_image):

    focal_length = (width_in_rf_image * Known_distance) / real_width
    return focal_length


def obj_data(img):
    obj_width = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    _, mask1 = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        x = 600
        if cv2.contourArea(c) > x:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            obj_width = w
    return obj_width


def Distance_finder(Focal_Length, Known_width, obj_width_in_frame):
    distance = (Known_width * Focal_Length) / obj_width_in_frame

    return distance


ref_image = cv2.imread("C:/Users/Dell/Desktop/SAUVC_2022/Distance/rf.png")
ref_image_obj_width = obj_data(ref_image)
Focal_length_found = Focal_Length_Finder(
    Known_distance, Known_width, ref_image_obj_width
)
cv2.imshow("ref_image", ref_image)

print(Focal_length_found)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    obj_width_in_frame = obj_data(frame)
    Distance = -1
    if obj_width_in_frame != 0:
        Distance = Distance_finder(Focal_length_found, Known_width, obj_width_in_frame)
        cv2.putText(
            frame,
            f"Distance: {round(Distance,2)} CM",
            (30, 35),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
    if Distance > 16:
        forward()
    else:
        right()

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
