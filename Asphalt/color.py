import cv2 
import numpy as np 

cam = cv2.VideoCapture(0)
cv2.namedWindow('Color Detection')

def window(x):
    pass

cv2.createTrackbar('Hue', 'Color Detection' , 0, 179, window)
cv2.createTrackbar('Saturation','Color Detection', 0 ,255,window)
cv2.createTrackbar('Value','Color Detection', 0,255,window)

while True:
    ret ,img = cam.read()
    img = np.flip(img ,axis =1)
    img =  cv2.resize(img ,(480 ,360))
    hsv = cv2.cvtColor(img ,cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (11,11),0)

    h = cv2.getTrackbarPos('Hue', 'Color Detection')
    s = cv2.getTrackbarPos('Saturation', 'Color Detection')
    v = cv2.getTrackbarPos('Value', 'Color Detection')

    lower_color = np.array([h ,s, v])
    upper_color = np.array([100,255,255])
    mask = cv2.inRange(hsv , lower_color,upper_color)
    cv2.imshow ('Color Detection', cv2.bitwise_and(img ,img ,mask=mask))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
#31 33 153