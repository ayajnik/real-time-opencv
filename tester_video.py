import cv2
import os
import numpy as np
import video as fr


#This module takes images  stored in diskand performs face recognition
test_img=cv2.imread('C:\\Users\\Yadvi\\PycharmProjects\\real_time\\training\\0\\frame0.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

#Comment belows lines when running this program second time.Since it saves training.yml file in directory
faces,faceID=fr.labels_for_training_data('C:\\Users\\Yadvi\\PycharmProjects\\real_time\\training')
face_recognizer=fr.train_classifier(faces,faceID)

name={0:"Ayush Yajnik"}#creating dictionary containing names for each label

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    print("confidence:",confidence)
    print("label:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if(confidence>37):#If confidence more than 37 then don't print predicted face text on screen
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,1000))
cv2.imshow("face dtecetion tutorial",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows()

##saving the data
face_recognizer.write('trainingData.yml')