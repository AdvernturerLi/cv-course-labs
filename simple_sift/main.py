import cv2
import numpy as np

from gaussian_pyramid import build_gaussian_pyramid, build_dog_pyramid, save_pyramid_images
from keypoint_detection import detect_keypoints
from descriptor import compute_gradient, assign_orientation, compute_descriptor
from utils import draw_keypoints


def main():

    img = cv2.imread("simple_sift\input\image.png", 0)

    # Step1 高斯金字塔
    gaussian_pyr = build_gaussian_pyramid(img)

    # Step2 DoG
    dog_pyr = build_dog_pyramid(gaussian_pyr)

    save_pyramid_images(gaussian_pyr, dog_pyr)

    # Step3 极值点检测
    keypoints = detect_keypoints(dog_pyr)

    draw_keypoints(img, keypoints)

    # Step4 描述子
    descriptors = []

    for (o,s,y,x) in keypoints:

        gimg = gaussian_pyr[o][s]

        mag, angle = compute_gradient(gimg)

        if y < 8 or x < 8 or y >= gimg.shape[0]-8 or x >= gimg.shape[1]-8:
            continue

        main_angle = assign_orientation(mag, angle, y, x)

        desc = compute_descriptor(mag, angle, y, x, main_angle)

        descriptors.append(desc)

    descriptors = np.array(descriptors)

    np.save("simple_sift\\output\\descriptors.npy", descriptors)

    print("Keypoints:", len(keypoints))
    print("Descriptors:", descriptors.shape)


if __name__ == "__main__":
    main()