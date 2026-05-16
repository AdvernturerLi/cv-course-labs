import numpy as np
import cv2

DESC_GRID = 4
DESC_BINS = 8
HALF_PATCH = 8


def compute_gradient(image):

    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(dx**2 + dy**2)
    angle = (np.degrees(np.arctan2(dy, dx)) % 360)

    return mag, angle


def assign_orientation(mag, angle, y, x):

    hist = np.zeros(36)
    bin_size = 10

    for i in range(-8,9):
        for j in range(-8,9):

            yy = y+i
            xx = x+j

            b = int(angle[yy,xx] / bin_size) % 36
            hist[b] += mag[yy,xx]

    main_angle = np.argmax(hist) * bin_size
    return main_angle


def compute_descriptor(mag, angle, y, x, main_angle):

    desc = np.zeros(128)
    cell = 4
    bin_size = 360 / DESC_BINS
    cos_a = np.cos(np.radians(-main_angle))
    sin_a = np.sin(np.radians(-main_angle))
    h, w = mag.shape

    for i in range(DESC_GRID):
        for j in range(DESC_GRID):
            hist = np.zeros(8)

            for di in range(cell):
                for dj in range(cell):
                    # 相对于关键点的偏移（旋转对齐）
                    oy = -HALF_PATCH + i*cell + di
                    ox = -HALF_PATCH + j*cell + dj
                    ry = int(round(y + cos_a*oy - sin_a*ox))
                    rx = int(round(x + sin_a*oy + cos_a*ox))

                    if ry < 0 or ry >= h or rx < 0 or rx >= w:
                        continue

                    ang = (angle[ry, rx] - main_angle) % 360
                    b = int(ang / bin_size) % 8
                    hist[b] += mag[ry, rx]

            idx = (i*DESC_GRID + j)*DESC_BINS
            desc[idx:idx+8] = hist

    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm

    return desc