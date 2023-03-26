import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep 
import numpy as np
from pynput.keyboard import Controller , Key

cap= cv2.VideoCapture(0)

#Window size
cap.set(3,1280)
cap.set(4,720)

detector =  HandDetector(detectionCon=0.8 , maxHands=2)
keys = [["Q", "W", "E", "R", "T", "Y","U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z","X","C","V","B", "N ", "M"]]

finaltext = ""

keyboard = Controller()
#Handles Drawing of all the buttons :
#Instead of writing one by one button make a list
def drawALL( img , buttonList):
    imgNew =  np.zeros_like(img,np.uint8)
    for button in buttonList:  
        x,y = button.pos
        cvzone.cornerRect(imgNew,(button.pos[0], button.pos[1],button.size[0], button.size[1]),20 ,rt=0)
        cv2.rectangle(imgNew ,button.pos,(x + button.size[0], y + button.size[1]),(155,15,0), cv2.FILLED )
        cv2.putText(imgNew, button.text, (x+40,
                    y+60), cv2.FONT_HERSHEY_PLAIN,2, (255,255,255),3)
    out = img.copy()
    alpha = -0.5
    mask = imgNew.astype(bool)

    out[mask] = cv2.addWeighted(img , alpha, imgNew ,1 - alpha , 0)[mask]

    return out
class Button():
    def __init__(self , pos , text , size=[85,85]):
        self.pos=pos
        self.size=size
        self.text=text
    
       
    
buttonList = []
for i in range(len(keys)):
    for j,key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50 , 100*i+50],key))

    


while True :
    success, img = cap.read()
    #Find hands

    hands, img = detector.findHands(img)
    bboxInfo= detector.findHands(img , draw=False)
    lmList= detector.findHands(img , draw=False) 
    img = drawALL(img,buttonList)
    
    #Check whether hand or not:

   
      
    if hands:
        # Hand 1
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 Landmarks points
        bbox = hand["bbox"]  # Bounding Box info x,y,w,h
        centerPoint = hand["center"]  # center of the hand cx,cy
        handType = hand["type"]  # Hand Type Left or Right

    if len(hands) == 2:
        hand2 = hands[1]
        lmList2 = hand2["lmList"]  # List of 21 Landmarks points
        bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
        centerPoint2 = hand2["center"]  # center of the hand cx,cy
        handType2 = hand2["type"]  # Hand Type Left or Right    
    
        if lmList:
            for button in buttonList:
                x,y = button.pos
                w,h = button.size


            #Tip of finger is point no.8(index finger)
                if x < lmList[8][0] < x+w and y<lmList[8][1]<y+h:
                        cv2.rectangle(img ,button.pos,(x + w, y + h),(175,0,175), cv2.FILLED )
                        cv2.putText(img , button.text, (x+20,y+65), cv2.FONT_HERSHEY_PLAIN,4, (255,255,255),4)
                        l, _ , _= detector.findDistance(centerPoint,centerPoint2, img)
                        



                        if l<1000:
                            # keyboard.press(button.text)
                            keyboard.press(button.text.lower() if len(button.text)==1 else Key.space) # Press lowercase letter or space
                            cv2.rectangle(img ,button.pos,(x + w, y + h),(0,255,0), cv2.FILLED )
                            cv2.putText(img , button.text, (x+20, y+65), cv2.FONT_HERSHEY_PLAIN,4, (255,255,255),4)
                            finaltext += button.text
                            sleep(1)

# ...



    cv2.rectangle(img ,(50,350),(700,450),(0,0,0), cv2.FILLED )
    cv2.putText(img , finaltext, (60,430), cv2.FONT_HERSHEY_PLAIN,4, (255,255,255),4)        
    
    cv2.imshow("image",img)

    cv2.waitKey(1)==27
