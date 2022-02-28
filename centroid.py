import cv2
import numpy as np
from matplotlib import pyplot as plt


def findCentroid(mask, DEBUG=False):
    image = cv2.imread('./refvos/images/inference/image/0000.jpg')

    # find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if DEBUG:
        img_cnt = image
        cv2.drawContours(img_cnt, contours, -1, (0,0,255), 2)
        plt.imshow(img_cnt)
        plt.show()

    for c in contours:
        # find minimum bounding rectangle
        rect = cv2.minAreaRect(c)

        # convert to 4 corner points
        box = cv2.boxPoints(rect)
        box = (np.round(box)).astype(int)
        if DEBUG:
            img_rect = image
            cv2.drawContours(img_rect, [box], -1, (0,0,255), 2)
            plt.imshow(img_cnt)
            plt.show()

        width = rect[1][0]
        height = rect[1][1]
        # principle angle
        angle = rect[2]
        if width < height:
            angle = -(90 - angle)

        # find centroid of the contour
        M = cv2.moments(c)
        if M["m00"] < 10:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # show the final results
        if DEBUG:
            img_out = image
            cv2.circle(img_out, (cX, cY), 7, (255, 0, 0), -1)
            plt.axline((cX,cY), slope=np.tan(np.radians(angle)), color='blue')
            plt.title(f'Centroid:{cX, cY}\nPrinciple angle:{angle:.2f} degree')
            plt.imshow(img_out)
            plt.show()

        return cX, cY, angle