import cv2
import mediapipe as mp
import numpy as np
import math
import random
def draw_curved_branch(img,p1,p2,color,thickness):
    mid_x=(p1[0] + p2[0]) // 2 +random.randint(-15,15)
    mid_y=(p1[1] + p2[1]) // 2 +random.randint(-15,15)

    pts=[]
    for t in np.linspace(0,1,15):
        x=(1-t)** 2 * p1[0] + 2 * (1-t) * t * mid_x + t**2 * p2[0]
        y=(1-t)** 2 * p1[1] + 2 * (1-t) * t * mid_y + t**2 * p2[1]
        pts.append([int(x),int(y)])

    pts=np.array(pts,np.int32).reshape((-1,1,2))
    cv2.polylines(img,[pts],False,color,thickness,cv2.LINE_AA)

def drawing_spider_lilly(img,center,radius):





mp_hands=mp.solutions.hands
hands=mp_hands.Hands(max_num_hands=20,min_detection_confidence=0.7)
mp_draw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)

while True:
    success,frame=cap.read()

    if not success:
        print("Camera Image Could Not Be Obtained")
        break

    frame=cv2.flip(frame,1)
    h,w,c=frame.shape

    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hands_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hands_landmarks,mp_hands.HAND_CONNECTIONS)

            landmarks=hands_landmarks.landmark

            is_index_open=landmarks[8].y < landmarks[6].y
            is_middle_open=landmarks[12].y < landmarks[10].y
            is_ring_open=landmarks[16].y < landmarks[14].y

            if is_index_open and is_middle_open and is_ring_open:
                palm_x=int(landmarks[9].x*w)
                palm_y=int(landmarks[9].y*h)

                fingertip_x=int(landmarks[8].x*w)
                fingertip_y=int(landmarks[8].y*h)

                wrist_x=int(landmarks[0].x*w)
                wrist_y=int(landmarks[0].y*h)

                cv2.circle(frame,(palm_x,palm_y),25,(0,0,255),-1)
                cv2.circle(frame,(fingertip_x,fingertip_y),15,(0,0,255),-1)
                cv2.circle(frame,(wrist_x,wrist_y),20,(0,0,255),-1)

    cv2.imshow("Tokyo Ghoul - Spider Lilly Vision",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





