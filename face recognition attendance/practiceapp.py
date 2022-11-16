import cv2
import numpy as np
import face_recognition

imagevinay = face_recognition.load_image_file("C:\\Users\\DELL\\Desktop\\ML Project 7th semester\\face recognition attendance\\students photos\\vinaypratyush.jpg")
imagevinay = cv2.cvtColor(imagevinay,cv2.COLOR_BGR2RGB)

imagevinaytest = face_recognition.load_image_file("C:\\Users\\DELL\\Desktop\\ML Project 7th semester\\face recognition attendance\\students photos\\vinaypratyushtest.jpg")
imagevinaytest = cv2.cvtColor(imagevinaytest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imagevinay)[0]
encodedVinay= face_recognition.face_encodings(imagevinay)[0]
cv2.rectangle(imagevinay,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imagevinaytest)[0]
encodedVinayTest= face_recognition.face_encodings(imagevinaytest)[0]
cv2.rectangle(imagevinaytest,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodedVinay],encodedVinayTest)
faceDistance=face_recognition.face_distance([encodedVinay],encodedVinayTest)
print(results,faceDistance)
cv2.putText(imagevinaytest,f'{results}{faceDistance[0]}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Vinay Pratyush',imagevinay)
cv2.imshow('Vinay Pratyush Test', imagevinaytest)

cv2.waitKey(0)

