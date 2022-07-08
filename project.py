import numpy as np
import cv2
import pyttsx3
engine = pyttsx3.init()
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
classes= []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)


cap = cv2.VideoCapture(0) #captureDevice = camera
cap1 = cv2.VideoCapture('Lightning Bolt At Night.mp4')
while True:
    ret, frame = cap.read()
    ret1, frame1 =cap1.read()
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    edges=cv2.Canny(blur,100,50)
    counter=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in counter[0]:
        area = cv2.contourArea(c)
        if area > 1000:
            x,y,w,h=cv2.boundingRect(c)
            cv2.drawContours(frame1, [c], -1, (0,255,0),2)
            cv2.putText(frame1,'lightning',(x+w,y+h),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            engine.say("lightning alert!")
            engine.runAndWait()



    cv2.imshow("frame",frame1)
    (class_ids, score, bboxes)= model.detect(frame)
    for class_id, score, bbox in zip(class_ids, score, bboxes):
        (x,y,w,h)=bbox
        class_name = classes[class_id[0]]
        cv2.putText(frame, class_name,(x,y-10),cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (200,0, 50), 3)
        if class_name == 'bird':
            engine.say("birds detected")
            engine.runAndWait() 

    cv2.imshow('my frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

