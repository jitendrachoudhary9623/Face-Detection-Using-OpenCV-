import cv2
import numpy as np


class FaceDetector():
    def __init__(self):
      
        self.faceCascade = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml")
    def detectFaces(self,image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        boundaries=self.detect(image)
        if len(boundaries) is 0:
            cv2.putText(image,"No Face Detected",(75,190),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(100,170,0),2)
            return image
        for (x,y,w,h) in boundaries:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,127),2)
        return image

		

    def detect(self, image, scaleFactor=1.3, minNeighbors=5, minSize=(300, 300)):

        boundaries = self.faceCascade.detectMultiScale(image,scaleFactor=scaleFactor,minNeighbors=minNeighbors,minSize=minSize,flags=cv2.CASCADE_SCALE_IMAGE)

        return boundaries

	
    		


		
