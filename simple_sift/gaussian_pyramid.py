import cv2
import numpy as np


NUM_OCTAVES = 3
NUM_SCALES = 5
SIGMA_BASE = 1.6
K = 2 ** (1.0 / (NUM_SCALES - 3))


def build_gaussian_pyramid(image):
    gaussian_pyr = []
    current = image.astype(np.float32)

    for o in range(NUM_OCTAVES):
        octave_imgs = []
        sigma = SIGMA_BASE

        for s in range(NUM_SCALES):
            blurred = cv2.GaussianBlur(current, (0, 0), sigma)
            octave_imgs.append(blurred)
            sigma *= K

        gaussian_pyr.append(octave_imgs)

        seed = octave_imgs[-3]
        current = cv2.resize(
            seed,
            (seed.shape[1] // 2, seed.shape[0] // 2),
            interpolation=cv2.INTER_NEAREST
        )

    return gaussian_pyr


def build_dog_pyramid(gaussian_pyr):

    dog_pyr = []

    for octave in gaussian_pyr:
        dogs = []
        for i in range(1, len(octave)):
            dogs.append(octave[i] - octave[i-1])
        dog_pyr.append(dogs)

    return dog_pyr


def save_pyramid_images(gaussian_pyr, dog_pyr):

    for o, octave in enumerate(gaussian_pyr):
        for s, img in enumerate(octave):

            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(f"output/gaussian/octave{o}_scale{s}.png", norm)

    for o, octave in enumerate(dog_pyr):
        for s, img in enumerate(octave):

            norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(f"output/dog/octave{o}_dog{s}.png", norm)