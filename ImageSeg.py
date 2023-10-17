import matplotlib.pyplot as plt
import numpy as np
import cv2


def kMeanSeg(img):
    
    #processing the image
    twoDimage = img.reshape((-1,3))
    twoDimage = np.float32(twoDimage)

    #defining critria

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 6
    attempts= 20

    #applying k-maeans

    ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))



    plt.axis('off')
    plt.imshow(result_image)

image = cv2.imread("London_Big_Ben_Phone_box.jpg")
#changing to RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow("this is color changed", img)

kMeanSeg(img)


cv2.waitKey(0)