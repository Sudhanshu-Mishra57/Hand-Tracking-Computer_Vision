# import modules
import mediapipe as mp
import cv2
import time


mpHands = mp.solutions.hands
hands = mpHands.Hands()

# for draw hand
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# caputure video
Cap = cv2.VideoCapture(0)

while True:
    success, img = Cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)

                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

            # show/draw landmark on hand if hand detected
            mpDraw.draw_landmarks(img, handlms,mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,78), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0),3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)
