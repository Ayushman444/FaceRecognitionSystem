import cv2
import numpy as np
from PIL import Image
import os
import time
from datetime import datetime
import pyttsx3, openpyxl


ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S") 

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2 )
        
        id, pred = clf.predict(gray_img[y:y+h,x:x+w])
        confidence = int(100*(1-pred/300))
        

        book = openpyxl.load_workbook('C:/Users/Lenovo/Desktop/FaceRecognitionSystem/Attendance/Attendance.XLSX')
        sheet = book.active
        L1 = []
        for row in sheet['1']:  
            L1.append(row.value)

        if date not in L1:
            sheet.cell(row=1,column = len(sheet['1'])+1).value = date
            book.save('C:/Users/Lenovo/Desktop/FaceRecognitionSystem/Attendance/Attendance.XLSX')


        if confidence>70:
            book = openpyxl.load_workbook('C:/Users/Lenovo/Desktop/FaceRecognitionSystem/Attendance/Attendance.XLSX')
            sheet = book.active
            L1 = []
            L2 = []
            for i in sheet['A']:
                L1.append(i.value)
            for j in sheet['B']:
                L2.append(j.value)
            ind = L1.index(id)
            name = L2[ind]
           
            
            cv2.putText(img, name, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)   
            
        else:
            cv2.putText(img, "UNKNOWN", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)
    
        L2= []
        for column in sheet['A']:
                L2.append(column.value)
            
        if int(id) in L2:
                x = L2.index(int(id))+1
                sheet.cell(row=x,column=len(sheet[str(x)])).value = 'p'
                book.save('C:/Users/Lenovo/Desktop/FaceRecognitionSystem/Attendance/Attendance.XLSX')
                
    book = openpyxl.load_workbook('C:/Users/Lenovo/Desktop/FaceRecognitionSystem/Attendance/Attendance.XLSX') 
    sheet = book.active
    L3 = []
    col = sheet['1']
    len_col = len(col)
    x = openpyxl.utils.cell.get_column_letter(len_col)
     
    for column in sheet[x]:
        L3.append(column.value)
    
    for i in range(len(L3)):
        if L3[i]==date or L3[i]=='p':
            pass
        elif L3[i]==None:
            sheet.cell(row=i+1, column = len_col).value = 'a'
            book.save('C:/Users/Lenovo/Desktop/FaceRecognitionSystem/Attendance/Attendance.XLSX')
        
        
    return img


faceCascade = cv2.CascadeClassifier("C:/Users/Lenovo/Desktop/FaceRecognitionSystem/haarcascade_frontalface_default.xml")

clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("C:/Users/Lenovo/Desktop/FaceRecognitionSystem/classifier.xml")

video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    img = draw_boundary(img, faceCascade, 1.3, 6, (255,255,255), "Face", clf)
    cv2.imshow("face Detection", img)
    
    if cv2.waitKey(1)==13:
        break
video_capture.release()
cv2.destroyAllWindows()