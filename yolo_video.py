import numpy as np
import argparse
import cv2

# object tracking
import 

threesixty_one = '192.168.0.12-798248343'  #  360 cam #1
threesixty_two = '192.168.0.11-209404134'  #  360 cam #2
ninety_only = '192.168.0.16-1804746112'    #   90 cam

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('data/192.168.0.12-798248343/07.14.2022/13h40m38s.vida')  # Use .vida format

# Stabil kamera, vi har kun objekter der ændre sig. 
object_dector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    # Brug højde og width til at snæve os ind op billede rammen.  
    height, width, _ = frame.shape
    #print(height, width)
    
    # Udtræk region
    
    roi = frame[1000: 2192, 1000: 2192]
    
    # mask gør alt vi får bevægelser som bliver hvidt mens det andet er sort.
    # Der er mange hvide elementer.
    mask = object_dector.apply(frame)
    # sæt et grænseværdie hvor pixe 254 er max bredde.  
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Alt der er hvidt omfarve vi med grønt det gør vi kan fjerne omrdåer
    for cnt in contours:
        # beregn  area og fjern små elementer
        area = cv2.contourArea(cnt)
        if area > 100: 
            #cv2.drawContours(frame, [cnt],-1, (0, 255, 0), 2)
            # træk box
            x, y, w, h = cv2.boundingRect(cnt)
            # koordinater for x w 
            cv2.rectangle(frame, (x,y), (x+w, y + h), (0, 255, 0), 3)
    
    cv2.imshow('Roi', roi)
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)
    
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()    

