# Relevant packages to run the model
import cv2
import numpy as np

# This network contains the YOLO weights, configuration files
neural_network = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Extracting object names for the model prediction
object_classes = []     # Loading names into a list
with open('coco.names', 'r') as f:
    object_classes = f.read().splitlines()

# Input from camera can be taken
cap = cv2.VideoCapture(0)
# img = cv2.imread('image.jpg')    //This can be used when input is an image


# The output shall be displayed with fonts and colors with prediction & accuracy rate
fonts = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

# Normalising the Input Image from pixel-to-pixel value and converting it from JPG to RGB format
    Normalizing_Layer = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # Setting the Input image from blob to the network
    neural_network.setInput(Normalizing_Layer)
    # Provides the bounding information and feature extraction from forward layers
    Output_Layer = neural_network.getUnconnectedOutLayersNames()
    layerOutputs = neural_network.forward(Output_Layer)

    bounding_boxes = []    # extract the bounding boxes of predicted objects
    prediction_percentage = []  # predicts the detected objects
    class_ids = []              # defines the object subjected to particular class

# To extract the boxes, confidence & classes information of input
    for output in layerOutputs:

         # used to extract the information from each of the output detection contains in each element with
         # 4 boundary box offset, 1 box confidence, and 80 predicted class information
         for detection in output:
            scores = detection[5:]

            # using the numpy argument max to extract the highest scores
            class_id = np.argmax(scores)

            # predicts the probability
            confidence = scores[class_id]

            if confidence > 0.2:
                # details of coordinates and size of the predicted object in normalised form
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                # the model predicts using centres of bounding boxes to extract the position with the help of OpenCV tool
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                bounding_boxes.append([x, y, w, h])
                prediction_percentage.append((float(confidence)))
                class_ids.append(class_id)

    #  nms_threshold is the IOU threshold used in non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(bounding_boxes, prediction_percentage, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = bounding_boxes[i]
            label = str(object_classes[class_ids[i]])
            confidence = str(round(prediction_percentage[i], 2))
            #color = "0.255.0"

            #cv2. rectangle() method is used to draw a rectangle on any image.
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

            # to put a Text boxes with desired class category
            cv2.putText(img, label + " " + confidence, (x, y+20), fonts, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()