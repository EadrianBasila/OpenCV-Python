#Hi
import cv2

img = cv2.imread('OpenCV\Eadrian.png')


classNames = []
classFile = 'OpenCV\coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'OpenCV\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'OpenCV\meh.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.5)
print(classIds,bbox)


cv2.imshow("Output",img)
cv2.waitKey(0)
