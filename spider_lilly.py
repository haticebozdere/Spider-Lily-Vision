import cv2
import mediapipe as mp
import numpy as np
import math
import random


def draw_curved_branch(img, p1, p2, color, thickness):
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2 + 25

    pts = []
    for t in np.linspace(0, 1, 15):
        x = (1 - t)**2 * p1[0] + 2 * (1 - t) * t * mid_x + t**2 * p2[0]
        y = (1 - t)**2 * p1[1] + 2 * (1 - t) * t * mid_y + t**2 * p2[1]
        pts.append([int(x), int(y)])

    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)


def drawing_spider_lilly(img, center, radius):
    pass


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=20, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

angle = 0
bloom = 0.0

while True:
    success, frame = cap.read()

    if not success:
        print("Camera Image Could Not Be Obtained")
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    angle = (angle + 1.5) % 360

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hands_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hands_landmarks.landmark

            thumb_x,thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)
            index_x,index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)

            dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
            bloom = np.clip((dist - 30 ) / 110,0.0,1.0)

            root_points = (w // 2 , h - 40)
            draw_curved_branch(frame,root_points,(index_x,index_y),(60,190,80),2)

            test_radius = int(10 + 40 * bloom)
            cv2.circle(frame, (index_x, index_y), test_radius, (0, 0, 255), -1)
    cv2.imshow("Tokyo Ghoul - Spider Lilly Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()