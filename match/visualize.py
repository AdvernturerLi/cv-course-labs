import cv2
import numpy as np


def draw_matches(img1, img2, kp1, kp2, matches):

    h1, w1 = img1.shape
    h2, w2 = img2.shape

    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)

    canvas[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for i, j in matches:

        x1, y1 = kp1[i]
        x2, y2 = kp2[j]

        cv2.line(
            canvas,
            (x1, y1),
            (x2 + w1, y2),
            (0, 255, 0),
            1
        )

        cv2.circle(canvas, (x1, y1), 2, (0, 0, 255), -1)
        cv2.circle(canvas, (x2 + w1, y2), 2, (255, 0, 0), -1)

    return canvas