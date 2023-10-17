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



    # plt.axis('off')
    # plt.imshow(result_image)
    cv2.imshow("sd", result_image)



def contourDetection(sample):
    img = cv2.resize(sample, (256,256))
    #plt.axis('off')
    #plt.imshow(img)

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)

    edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256,256), np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    cv2.imshow("sd", segmented)
    # plt.axis('off')
    # plt.imshow(thresh)






image = cv2.imread("London_Big_Ben_Phone_box.jpg")
#image = cv2.imread("ball.webp")
#changing to RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#cv2.imshow("this is color changed", img)

#kMeanSeg(img)
contourDetection(img)


cv2.waitKey(0)