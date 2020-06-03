# import the necessary packages
from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
from cv2 import cv2


def detect_people(frame, net, ln, personIdx=0):
    # grab the dimensions of the frame and initialise the list of results
    """
    frame: the frame from your video file or directly from your webcam
    net:   The pre-initialized and pre-trained YOLO object detection model
    ln:    The YOLO CNN output layer names
    personIdx: The YOLO model can detect many types of object; this index is 
            specifically for the person class, as we don't be considering other pbjects
    """

    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward pass of the YOLO object
    # detector, giving us our bounding boxes and associated probabilities

    blob = cv2.dnn.blobFromImage(frame,
                                 1 / 255.0, (416, 416),
                                 swapRB=True,
                                 crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    #initialise our lists of detected bounding boxes, centroids, and confidences
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        #loop over each of detections
        for detection in output:
            #extract the clas ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter detections by (1) ensuring that the object detected was a person
            # (2) that the minium confidence is met.
            if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinated back relative to the size of the image,
                # keeping in mind that YOLO actually returns the center (x,y) of the bounding box
                # followed by boxes' width and height.
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x,y) to derive the top and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, centeroids, and confidence
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idx = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if (len(idx) > 0):
        for i in idx.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # update our results list to consist of the person prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results
