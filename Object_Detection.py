import cv2
import numpy as np
import smtplib
import playsound
import threading

Alarm_Status = False
net = cv2.dnn.readNet('yolov3_training_2000.weights', 'yolov3_testing.cfg')


def play_alarm_sound_function():
	while True:
		playsound.playsound('alarm-sound.mp3',True)

classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('RT-2.mp4')


font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    #scale_percent = 40 # percent of original size
    #width = int(img.shape[1] * scale_percent / 100)
    #height = int(img.shape[0] * scale_percent / 100)
    #dim = (width, height)
    
    #resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    height, width,_ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                if(class_id==0 ):
                    if(Alarm_Status==False):
                        threading.Thread(target=play_alarm_sound_function).start()
                        Alarm_Status=True
                        print("ID")
                        print(class_id)
                else:
                    Alarm_Status=False
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            for i in range(len(class_ids)):
                if(class_ids[i]==0 or class_ids[i]==2):
                    cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255), 2)
                    if(Alarm_Status==False):
                        threading.Thread(target=play_alarm_sound_function).start()
                        Alarm_Status=True
                        
                    else:
                        Alarm_Status=False
                elif(class_ids[i]==1 or class_ids[i]==3):
                    cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 2)
                    print(class_ids)

            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            print(class_ids)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()