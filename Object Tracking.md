# Object Tracking

Object tracking is a computer vision task that refers to the process of finding & tracking the position of a predefined object that is moving in the frames of a video.

## Object Tracking vs Object Detection

At times beginners confuse object tracking with object detection and use the two words interchangeably. But there is a slight difference between the two –

In the object detection task, we identify the object in a specific frame or a scene that may be just a static image. Whereas in object tracking we track the object which is in continuous motion in a video. In fact, if we perform object detection on every frame of the video its resulting effect is of object tracking only.

## Types of Object Tracking Algorithms

### Single Object Tracking

Single object tracking refers to the process of selecting a region of interest (in the initial frame of a video) and tracking the position (i.e. coordinates) of the object in the upcoming frames of the video. We will be covering some of the algorithms used for single object tracking in this article.

### Multiple Object Tracking (MOT)

Multiple object tracking is the task of tracking more than one object in the video. In this case, the algorithm assigns a unique variable to each of the objects that are detected in the video frame. Subsequently, it identifies and tracks all these multiple objects in consecutive/upcoming frames of the video.

Since a video may have a large number of objects, or the video itself may be unclear, and there can be ambiguity in direction of the object’s motion Multiple Object Tracking is a difficult task and it thus relies on single frame object detection.

## What makes Object Tracking difficult


### 

## Libraries

### Installing OpenCV

We install the opencv-contrib-python library for our purpose. It is a different community maintained OpenCV Python package that contains some extra features and implementation than the regular OpenCV Python package.

```python
pip install opencv-contrib-python
```

If you don´t have the right version then  use the following command and install the right oen:

```python
pip uninstall opencv-contrib-python opencv-python opencv-contrib-python-headless && pip install opencv-contrib-python
```

## Algoritmes

### [KCF Object Tracking](https://arxiv.org/pdf/1404.7584.pdf)

KCF stands for Kernelized Correlation Filter, it is is a combination of techniques of two tracking algorithms (BOOSTING and MIL tracker). It is supposed to translate the bounding box (position of the object) using circular shift. In simple words, the KCF tracker focuses on the direction of change in an image(could be motion, extension or, orientation) and tries to generate the probabilistic position of the object that is to be tracked

The KCF object tracking is implemented in the `TrackerKCF_create()` module of OpenCV python. Below is the code along with the explanation

```python
tracker = cv2.TrackerKCF_create()
video = cv2.VideoCapture('data/192.168.0.12-798248343/07.14.2022/13h40m38s.vida')  # Use .vida format
ok,frame=video.read()
bbox = cv2.selectROI(frame)
ok = tracker.init(frame,bbox)
while True:
   ok,frame=video.read()
   if not ok:
        break
   ok,bbox=tracker.update(frame)
   if ok:
        (x,y,w,h)=[int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
   else:
        cv2.putText(frame,'Error',(100,0),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
   cv2.imshow('Tracking',frame)
   if cv2.waitKey(1) & 0XFF==27:
        break
cv2.destroyAllWindows()
```

Line 1-3: We first initialize the ‘KCF’ tracker object. Next, we initialize the video and then use the ‘read()’ function to fetch the first frame of the video.

Line 5: We initialize the ‘selectROI’ function with the first frame of the video which we fetched on the second line and store its value in the ‘bbox’ variable.

Line 7: We initialize the tracker (using ‘init’) with the frame (in which we selected our region of interest) and position (bbox) of the object to be tracked.

Line 9: Initialize a while loop that loops through the frames of our video.

Line 10: Use the ‘read()’ function on the video object to fetch the frames of the video along with a flag parameter(‘ok’) which informs if the frame fetching process was successful or not.

Line 11-12: If the flag parameter is false the execution stops i.e. if the video is not fetched properly the execution is stopped.

Line 13: We use the tracker ‘update’ function to pass a new consecutive frame with every iteration of the loop. It returns two variables, first is a flag parameter that informs if the tracking process was successful or not and the second returns the position of the tracked object in the frame if and only if the first parameter was true.

Line 14-16: If the ‘ok’ flag is true this block is executed. We fetched the position of the object in the ‘bbox’ variable, here we initialize the x,y coordinates and the values of width and height. Next, we use the OpenCV ‘rectangle’ function to put a bounding box around the detected object in consecutive frames of the video.

Line 17-18: If the tracker is unable to track the selected ROI or faces any errors, this block of code prints ‘Error’ on the video frames.

Line 19: Showing the video frames on a separate window using the ‘cv2.imshow’ function.

Line 20-21: If the user clicks the ‘escape’ button execution stops.

Line 22: Use the OpenCV ‘destroyAllWindows()’ function to close all lingering windows if there are any

When going though the other algoritme I will only mention those and not implement them.

### [GOTURN](https://cv-tricks.com/object-tracking/quick-guide-mdnet-goturn-rolo/)

###   [TLD (Tracking Learning Detection) Tracker](http://vision.stanford.edu/teaching/cs231b_spring1415/papers/Kalal-PAMI.pdf) 




## Litteratur

* https://broutonlab.com/blog/opencv-object-tracking
* https://docs.opencv.org/4.x/d2/d0a/tutorial_introduction_to_tracker.html
* https://blog.roboflow.com/object-tracking-how-to/
* https://blog.roboflow.com/zero-shot-object-tracking/
* https://viso.ai/deep-learning/object-tracking/