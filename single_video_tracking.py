import numpy as np
import argparse
import cv2

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[5]
tracker = cv2.TrackerMIL_create()
# object tracking

threesixty_one = '192.168.0.12-798248343'  #  360 cam #1
threesixty_two = '192.168.0.11-209404134'  #  360 cam #2
ninety_only = '192.168.0.16-1804746112'    #   90 cam

# Create a VideoCapture object and read from input file
video = cv2.VideoCapture('data/192.168.0.12-798248343/07.14.2022/13h40m38s.vida')  # Use .vida format
ret, frame = video.read()

frame_height, frame_width = frame.shape[:2]
# Resize the video for a more convinient view
frame = cv2.resize(frame, [frame_width//2, frame_height//2])
# Initialize video writer to save the results

output = (
    cv2
    .VideoWriter(f'{tracker_type}.avi', 
                         cv2
                         .VideoWriter_fourcc(*'XVID'), 
                         60.0, 
                         (frame_width//2,
                          frame_height//2), 
                         True)
)
if not ret:
    print('cannot read the video')
# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)
ret = tracker.init(frame, bbox)
# Start tracking
while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, [frame_width//2, frame_height//2])
    if not ret:
        print('something went wrong')
        break
    timer = cv2.getTickCount()
    ret, bbox = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    if ret:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else:
        cv2.putText(frame, "Tracking failure detected", (100,80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
    cv2.putText(frame, tracker_type + " Tracker", (100,20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
    cv2.imshow("Tracking", frame)
    
    output.write(frame)
    
    k = cv2.waitKey(1) & 0xff
    
    if k == 27 : break
        
video.release()
output.release()
cv2.destroyAllWindows()