
import cv2
import numpy as np 
# Image File Import and Resizing
#imgRaw = cv2.imread('OpenCV\Eadrian.png')
#img = cv2.resize(imgRaw, (350,350))

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
img_counter = 0
frm=1

# Standard Config Procedure
classNames = []
classFile = 'OpenCV\coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

threshold = 0.7
configPath = 'OpenCV\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'OpenCV\meh.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def stackImages(scale,imgArray):   #Function for Image Stacking
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold= threshold)   #Sets Confidence Level to 70%
    print(classIds,bbox) 

 
    if len(classIds) !=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img,box,color=(51, 255, 195),thickness=2)                                                                             # Draw Box on object being identified
            cv2.putText(img,"Object: " + classNames[classId-1], (box[0]+10,box[1]+20), cv2.FONT_HERSHEY_PLAIN,1,(51, 255, 195),thickness=2)     # Place Identifier on Object || classNames[classId-1]

            cv2.rectangle(img,box,color=(51, 255, 195),thickness=2)                                                                              # Draw Box on object being identified
            cv2.putText(img,"Confidence: " +str(round(confidence*100,2)) + "%", (box[0]+10,box[1]+40), cv2.FONT_HERSHEY_PLAIN,1,(51, 255, 195),thickness=2)    # Place Identifier on Object

            if  (classNames[classId-1]) == "person":
                img_name = "personDetected_{}.png".format(img_counter)
                cv2.imwrite(img_name, img)
                print("{} written!".format(img_name))
                img_counter += 1  
                cv2.imshow("Person Detected",img)
                break
    
    cv2.imshow("Output",img)
    cv2.waitKey(1)


  

   