import cv2
import sys
import numpy as np

sys.path.insert(0, '.')
from simple_sift.gaussian_pyramid import build_gaussian_pyramid, build_dog_pyramid
from simple_sift.keypoint_detection import detect_keypoints
from simple_sift.descriptor import compute_gradient, assign_orientation, compute_descriptor

img = cv2.imread('match/input/pic1.png', 0)
gauss_pyr = build_gaussian_pyramid(img)
dog_pyr = build_dog_pyramid(gauss_pyr)
keypoints = detect_keypoints(dog_pyr)

print('img shape:', img.shape)
print('total keypoints:', len(keypoints))

octave_grads = [compute_gradient(gauss_pyr[o][0]) for o in range(len(gauss_pyr))]
descs = []

for (o, s, y, x) in keypoints:
    mag, angle = octave_grads[o]
    oh, ow = mag.shape
    if x < 8 or x > ow - 9 or y < 8 or y > oh - 9:
        continue
    main_angle = assign_orientation(mag, angle, y, x)
    desc = compute_descriptor(mag, angle, y, x, main_angle)
    descs.append(desc)

arr = np.array(descs)
print('valid descs:', len(arr))
norms = np.linalg.norm(arr, axis=1)
print('zero descs:', np.sum(norms < 1e-6))

dists = []
for i in range(min(30, len(arr))):
    for j in range(i + 1, min(30, len(arr))):
        dists.append(np.linalg.norm(arr[i] - arr[j]))

print('avg inter-desc dist:', round(np.mean(dists), 4))
print('min inter-desc dist:', round(np.min(dists), 4))
