# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
import shutil

# determine only the *output* layer names that we need from YOLO
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def clean_up():

    if os.path.isdir('person'):
        shutil.rmtree('person')

    os.makedirs('person')

    if os.path.isdir('helmet'):
        shutil.rmtree('helmet')

    os.makedirs('helmet')

    if os.path.isdir('nonhelmet'):
        shutil.rmtree('nonhelmet')

    os.makedirs('nonhelmet')
    

clean_up()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# derive the paths to the YOLO weights and model configuration
personWeight = os.path.sep.join([args["yolo"], "yolov3_person.weights"])
helmetWeight = os.path.sep.join([args["yolo"], "yolov3_helmet.weights"])
personConfigPath = os.path.sep.join([args["yolo"], "yolov3_person.cfg"])
helmetConfigPath = os.path.sep.join([args["yolo"], "yolov3_helmet.cfg"])
 
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
def crop_image(weight, cfg, image):
    net = cv2.dnn.readNetFromDarknet(cfg, weight)

# load our input image and grab its spatial dimensions
    (H, W) = image.shape[:2]
 
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(get_output_layers(net))
 
# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

# loop over each of the layer outputs
    for output in layerOutputs:
    # loop over each of the detections
        for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
 
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
            if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
 
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    return (boxes, confidences, classIDs)

image = cv2.imread(args['image'])

(boxes, confidences, classIDs) = crop_image(personWeight, personConfigPath, image)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])

persons = []
# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
    # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        persons.append(image[y: y + h, x: x + w])

j = 0            
for person in persons:
    (boxes, confidences, classIDs) = crop_image(helmetWeight, helmetConfigPath, person)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])

    if len(idxs) > 0:
        cv2.imwrite(os.path.join('helmet',str(j) + '.jpg'), person)
    else:
        cv2.imwrite(os.path.join('nonhelmet',str(j) + '.jpg'), person)
    j += 1
    

'''# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
    args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
 
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)'''

'''helmetWeight = os.path.sep.join([args["yolo"], "yolov3_helmet.weights"])

net = net = cv2.dnn.readNetFromDarknet(configPath, helmetWeight)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

k=0

for person in boxes:
    person_image = image[person[1]:person[1]+person[3], person[0]:person[0]+person[2]]
    
    (H, W) = person_image.shape[:2]
 
# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
 
# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
    hboxes = []
    hconfidences = []
    hclassIDs = []

# loop over each of the layer outputs
    for output in layerOutputs:
    # loop over each of the detections
            for detection in output:
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
 
        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
                    if confidence > args["confidence"]:
            # scale the bounding box coordinates back relative to the
            # size of the image, keeping in mind that YOLO actually
            # returns the center (x, y)-coordinates of the bounding
            # box followed by the boxes' width and height
                            box = detection[0:4] * np.array([W, H, W, H])
                            (centerX, centerY, width, height) = box.astype("int")
 
            # use the center (x, y)-coordinates to derive the top and
            # and left corner of the bounding box
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

            # update our list of bounding box coordinates, confidences,
            # and class IDs
                            hboxes.append([x, y, int(width), int(height)])
                            hconfidences.append(float(confidence))
                            hclassIDs.append(classID)
    for i in range(len(hboxes)):
        
        (x, y) = (hboxes[i][0], hboxes[i][1])
        (w, h) = (hboxes[i][2], hboxes[i][3])
 
    # draw a bounding box rectangle and label on the image
        color = [int(c) for c in COLORS[hclassIDs[i]]]
        cv2.rectangle(person_image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[hclassIDs[i]], hconfidences[i])
        cv2.putText(person_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 2)

        cv2.imwrite("image"+str(k)+".png", person_image)'''
 
# show the output image
#cv2.imwrite("image.png", image)
#cv2.waitKey(0)
