# Object Detection with OpenCV and Python

## Introduction
---
Hello! In this tutorial I will show how to implement simple object detection using Python and OpenCV.

## What is a pixel?

All images consist of pixels which are the raw building blocks of images. Images are made of pixels in a grid. A 640 x 480 image has 640 columns (the width) and 480 rows (the height). There are 640 * 480 = 307200 pixels in an image with those dimensions.

Each pixel in a grayscale image has a value representing the shade of gray. In OpenCV, there are 256 shades of gray — from 0 to 255. So a grayscale image would have a grayscale value associated with each pixel.

Pixels in a color image have additional information. There are several color spaces that you’ll soon become familiar with as you learn about image processing. For simplicity let’s only consider the RGB color space.

## Setting up YOLO with Darknet
---
At first we need to have the model setup in the correct way. To do this we need to explain what Yolo is.

### What is YOLO?

YOLO (You Only Look Once) is a single stage detector that is popular due to it's speed and accuracy, it has been used in various applications to detect people, animals, traffic signals etc.

There are other object detectors such as R-CNN however, they are not as reliable as YOLO.

In this tutorial we will be building YOLOv3, v3 is significantly larger than previous models but it is the best one to use.

### What is Darknet?

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

### Setting up YOLO with Darknet

First we need to install darknet

```bash
git clone https://github.com/pjreddie/darknet && cd darknet
make
```

Next we need to download the pre-trained weights

```bash
wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/yolov3-tiny.weights
```

Now we have gotten the files we can turn to our project.

## Setting up the project
---
First we need to add the following files into the "weights" directory:

- coco.names
- yolov3.cfg
- yolov3.weights

Also we need to initialize the virtual environment:

```bash
python -m venv env
source env/bin/activate

mkdir weights
cp [darknet directory]/cfg/coco.names weights/
cp [darknet directory]/cfg/yolov3.cfg weights/
cp [darknet directory]/yolov3.weights
```

## Installing the dependencies
---
Next we need to install the dependencies needed for the example, create a "requirements.txt" file and then run activate.sh:

```bash
pip install -r requirements.txt
source env/bin/activate
```

## Writing the source code
---
Declare the necessary variables and initialize the network model:

```python
LABELS_FILE = "weights/coco.names"
CONFIG_FILE = "weights/yolov3.cfg"
WEIGHTS_FILE = "weights/yolov3.weights"
CONFIDENCE_THRESHOLD = 0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size = (len(LABELS), 3), dtype = "uint8")

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
```

The following function loops through the detected objects found in the image, checks to see if the confidence is above the minimal threshold and if so adds the box into the boxes array along with the coordinates the detection was discovered.

It then checks to make sure there is more than one detection and if so it draws the box along with the object label and confidence onto the image.

Finally the modified image is then shown on screen.

```python
def drawBoxes (image, layerOutputs, H, W):
  boxes = []
  confidences = []
  classIDs = []

  for output in layerOutputs:
    for detection in output:
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]

      if confidence > CONFIDENCE_THRESHOLD:
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)

  idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

  # Ensure at least one detection exists
  if len(idxs) > 0:
    for i in idxs.flatten():
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

      color = [int(c) for c in COLORS[classIDs[i]]]

      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  # Display the image
  cv2.imshow("output", image)
```

The next function reads an image file from the provided path, creates a blob from the image and sets the network input.

We then get the layer outputs and then pass the necessary variables to the function that was defined above.


```python
def detectObjects (imagePath):
  image = cv2.imread(imagePath)
  (H, W) = image.shape[:2]

  ln = net.getLayerNames()
  ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
  net.setInput(blob)
  layerOutputs = net.forward(ln)
  drawBoxes(image, layerOutputs, H, W)
```

Finally we of course need the main function, we use argparse to read the file path from the command line, call the above function and then wait for the user to press any key.

Once done we clean up by destroying the window.

```python
if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required = True, help = "Path to input file")

  args = vars(ap.parse_args())
  detectObjects(args["image"])

  cv2.waitKey(0)
  cv2.destroyAllWindows()
```

## Executing the program
---
The program can be executed by the following command:

```bash
python main.py -i 'picture_path.jpg'
```

If all goes well you should see the below image displayed:

---

![Sample](cowdet.png)
