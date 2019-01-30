# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 21:45:59 2019

@author: Mayank
"""

import time 
import cv2
from keras.models import load_model
import numpy as np


class_names = ['HELLO', 'MY','FRIEND']
width = 96
height = 96

# get the reference to the webcam
camera = cv2.VideoCapture(0)
camera_height = 500

model = load_model('GestureModel1.h5')

last_pred=-1
start_time=0
text=''
    
while(True):
    # read a new frame
    _, frame = camera.read()
    
    # flip the frameq
    frame = cv2.flip(frame, 1)

    # rescaling camera output
    aspect = frame.shape[1] / float(frame.shape[0])
    res = int(aspect * camera_height) # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    # add rectangle
    cv2.rectangle(frame, (300, 75), (650, 425), (240, 100, 0), 2)

    # get ROI
    roi = frame[75+2:425-2, 300+2:650-2]
    
    # parse BRG to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # resize
    roi = cv2.resize(roi, (width, height))
    
    # predict!
    roi_X = np.expand_dims(roi, axis=0)

    predictions = model.predict(roi_X)
    type_1_pred, type_2_pred,type_3_pred= predictions[0]
    print(type_1_pred)
    print(type_2_pred)
    print(type_3_pred)
   
    # add text
    if type_1_pred>0.9:
        print("hello")
        if last_pred!=0:
            start_time=0;
            start_time=time.time()
        else:
            end_time=time.time()
            elapsed_time=end_time-start_time
            if elapsed_time>2:
                text+=class_names[0]+' '
                cv2.putText(frame, text, (70, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
                start_time=0
        last_pred=0
    elif type_2_pred>0.9:
        print("MY")
        if last_pred!=1:
            start_time=0;
            start_time=time.time()
        else:
            end_time=time.time()
            elapsed_time=end_time-start_time
            if elapsed_time>2:
                text+=class_names[1]+' '
                cv2.putText(frame, text, (70, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
                start_time=0
        last_pred=1
    else:
        if type_3_pred>0.9:
            if last_pred!=0:
                start_time=0;
                start_time=time.time()
            else:
                end_time=time.time()
                elapsed_time=end_time-start_time
                if elapsed_time>2:
                    text+=class_names[0]+' '
                    cv2.putText(frame, text, (70, 170), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
                    start_time=0
            last_pred=0
            
        
        
        
        

    
    # show the frame
    cv2.imshow("Test out", frame)

    key = cv2.waitKey(1)

    # quit camera if 'q' key is pressed
    if key & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
model.summary()