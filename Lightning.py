import cv2
class_name="Lightning"
import pyttsx3
engine = pyttsx3.init()

cap=cv2.VideoCapture('Lightning Bolt At Night.mp4')
algo =  cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame =cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    edges=cv2.Canny(blur,100,50)
    counter=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in counter[0]:
        area = cv2.contourArea(c)
        if area > 1000:
            x,y,w,h=cv2.boundingRect(c)
            cv2.drawContours(frame, [c], -1, (0,255,0),2)
            cv2.putText(frame,'lightning',(x+w,y+h),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            engine.say("lightning alert!")
            engine.runAndWait()



        



    



    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
