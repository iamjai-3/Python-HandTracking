import cv2
import mediapipe as mp
import time


class  handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionConf = 0.5, trackConf = 0.5):
        self.mode = mode 
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf 
                
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):                
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Displays Hands recognized from camera with connections. 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:      
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                   
        return img

    def findPosition(self, img, handNo = 0, draw = True):
        
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,0), cv2.FILLED)

        return lmList
                
def main():
    prevTime = 0
    currTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img =  detector.findHands(img)
        lmList =  detector.findPosition(img)
        
        if len(lmList) != 0:
            print(lmList[4])

        currTime = time.time()
        fps =  1/(currTime-prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()