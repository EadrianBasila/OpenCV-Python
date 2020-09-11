#Hi
import cv2

# Image File Import and Resizing
imgRaw = cv2.imread('OpenCV\Eadrian.png')
img = cv2.resize(imgRaw, (350,350))

# Object Identifiers
classNames = []
classFile = 'OpenCV\coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Standard Config Procedure
configPath = 'OpenCV\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'OpenCV\meh.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)   #Sets Confidence Level to 50%
print(classIds,bbox) 

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img,box,color=(51, 255, 195),thickness=2)                                                               # Draw Box on object being identified
    cv2.putText(img,classNames[classId-1], (box[0]+10,box[1]+20), cv2.FONT_HERSHEY_PLAIN,1,(51, 255, 195),thickness=2)    # Place Identifier on Object

cv2.imshow("Output",img)
cv2.waitKey(0)
cv2.destroyAllWindows()