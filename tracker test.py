import numpy as np
import argparse
import cv2

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[5]
tracker = cv2.TrackerMIL_create()
import imutils
#vid = cv2.VideoCapture("sample1.mp4")
vid = cv2.VideoCapture('data/192.168.0.12-798248343/07.14.2022/13h40m38s.vida')  # Use .vida format
initial_box = None
tracker = cv2.TrackerMIL_create()
while True:
    fr = vid.read()
    if fr is None:
        break
    if initial_box is not None:
        (success, box) = tracker.update(fr)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(fr, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        fps.update()
        fps.stop()
        information = [
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.3f}".format(fps.fps())),]
        for (i, (k, v)) in enumerate(information):
            text = "{}: {}".format(k, v)
            cv2.putText(fr, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Output Frame", fr)
        key = cv2.waitKey(1) & 0xFF
    initial_box = cv2.selectROI("fr", fr, fromCenter=False,showCrosshair=True)
    tracker.init(fr, initial_box)
    fps = FPS().start()

cv2.destroyAllWindows()