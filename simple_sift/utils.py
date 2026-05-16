import cv2
import numpy as np


def draw_keypoints(image, keypoints):

    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for o,s,y,x in keypoints:

        scale = 2**o
        px = int(x*scale)
        py = int(y*scale)

        cv2.circle(vis, (px,py), 3, (0,255,0), 1)

    cv2.imwrite("output/keypoints.png", vis)