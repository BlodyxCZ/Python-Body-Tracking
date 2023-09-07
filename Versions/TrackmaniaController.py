import cv2
import mediapipe as mp
import keyboard

cap = cv2.VideoCapture(0)
cv2.setUseOptimized(True)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


def draw_screen():
    cv2.rectangle(img, (0,0), (640,480), (124,124,124), -1)
    cv2.rectangle(img, (1,0), (277,260), (255,0,0), 3)
    cv2.putText(img, 'LEFT', (115, 140), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,0,0), 2)
    cv2.rectangle(img, (280,0), (360,260), (0,255,0), 3)
    cv2.putText(img, 'FORWD', (295, 140), cv2.FONT_HERSHEY_PLAIN, 1.0,(0,255,0), 2)
    cv2.rectangle(img, (363,0), (639,260), (0,0,255), 3)
    cv2.putText(img, 'RIGHT', (475, 140), cv2.FONT_HERSHEY_PLAIN, 1.0,(0,0,255), 2)
    cv2.rectangle(img, (1,263), (639,480), (255,0,255), 3)
    cv2.putText(img, 'BRAKE', (295, 380), cv2.FONT_HERSHEY_PLAIN, 1.0,(255,0,255), 2)


def drive(cx, cy):
    keyboard.press('w')
    if cy > 260:
        keyboard.press('space')
    else:
        if cx > 360:
            keyboard.press('d')
            keyboard.release('a')
            keyboard.release('space')
        elif cx < 280:
            keyboard.press('a')
            keyboard.release('d')
            keyboard.release('space')
        else:
            keyboard.release('space')
            keyboard.release('a')
            keyboard.release('d')


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    draw_screen()

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(img, str(id), (cx+10,cy+10), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,0), 2)
                if id == 8:
                    cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                    drive(cx, cy)

    cv2.imshow('Image', img)
    cv2.setWindowProperty('Image', cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)
