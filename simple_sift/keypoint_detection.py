import numpy as np

CONTRAST_THRESH = 0.04
BORDER = 5


def detect_keypoints(dog_pyr):

    keypoints = []

    for o, dogs in enumerate(dog_pyr):

        for s in range(1, len(dogs)-1):

            prev = dogs[s-1]
            curr = dogs[s]
            nxt = dogs[s+1]

            h, w = curr.shape

            for y in range(BORDER, h-BORDER):
                for x in range(BORDER, w-BORDER):

                    val = curr[y, x]

                    if abs(val)/255.0 < CONTRAST_THRESH:
                        continue

                    cube = np.stack([
                        prev[y-1:y+2, x-1:x+2],
                        curr[y-1:y+2, x-1:x+2],
                        nxt[y-1:y+2, x-1:x+2]
                    ])

                    flat = cube.flatten()
                    others = np.delete(flat, 13)

                    if (val >= np.max(others)) or (val <= np.min(others)):
                        keypoints.append((o, s, y, x))

    return keypoints