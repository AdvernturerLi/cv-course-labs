import numpy as np
import cv2


def ransac_filter(kp1, kp2, matches, thresh=3.0):
    """
    kp1, kp2: [(x,y)]
    matches: [(i,j)]
    """

    if len(matches) < 4:
        return None, []

    pts1 = np.float32([kp1[i] for i, j in matches])
    pts2 = np.float32([kp2[j] for i, j in matches])

    H, mask = cv2.findHomography(
        pts1, pts2,
        cv2.RANSAC,
        thresh
    )

    inliers = []
    for k, m in enumerate(matches):
        if mask[k]:
            inliers.append(m)

    return H, inliers