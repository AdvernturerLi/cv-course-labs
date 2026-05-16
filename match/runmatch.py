import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ransac import ransac_filter
from visualize import draw_matches


def extract_features(img):
    sift = cv2.SIFT_create()
    kps_cv, descs = sift.detectAndCompute(img, None)
    if descs is None:
        return np.empty((0, 128), dtype=np.float32), []
    kps = [(int(kp.pt[0]), int(kp.pt[1])) for kp in kps_cv]
    return descs.astype(np.float32), kps


def match_descriptors(desc1, desc2, ratio):
    if len(desc1) == 0 or len(desc2) == 0:
        return []
    sim = desc1 @ desc2.T
    dist = np.sqrt(np.maximum(2.0 - 2.0 * sim / (
        np.linalg.norm(desc1, axis=1, keepdims=True) *
        np.linalg.norm(desc2, axis=1, keepdims=True).T + 1e-12
    ), 0.0))
    idx = np.argsort(dist, axis=1)
    nn1, nn2 = idx[:, 0], idx[:, 1]
    rows = np.arange(len(desc1))
    d1, d2 = dist[rows, nn1], dist[rows, nn2]
    d2_safe = np.where(d2 > 1e-12, d2, 1e-12)
    keep = d1 < ratio * d2_safe
    fwd = {int(i): int(nn1[i]) for i in np.where(keep)[0]}
    nn_back = np.argmin(dist, axis=0)
    return [(i, j) for i, j in fwd.items() if int(nn_back[j]) == i]


def run(img1, img2, ratio):
    desc1, kp1 = extract_features(img1)
    desc2, kp2 = extract_features(img2)
    print(f"  kp1={len(kp1)}, kp2={len(kp2)}")
    matches = match_descriptors(desc1, desc2, ratio)
    if len(matches) < 4:
        print(f"  ratio={ratio}: raw matches={len(matches)}, too few")
        return None, [], kp1, kp2, desc1, desc2
    H, inliers = ransac_filter(kp1, kp2, matches)
    rate = len(inliers) / len(matches) * 100 if matches else 0
    print(f"  ratio={ratio}: raw matches={len(matches)}, inliers={len(inliers)}, inlier_rate={rate:.2f}%")
    return H, inliers, kp1, kp2, desc1, desc2


def main():
    img1 = cv2.imread("match/input/pic1.png", 0)
    img2 = cv2.imread("match/input/pic2.png", 0)
    if img1 is None or img2 is None:
        print("Image not found!")
        return

    print("=== NNR threshold study ===")
    for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
        run(img1, img2, r)

    best_ratio = 0.7
    print(f"\n=== Final result (ratio={best_ratio}) ===")
    H, inliers, kp1, kp2, desc1, desc2 = run(img1, img2, best_ratio)

    if len(inliers) == 0:
        print("No inliers.")
        return

    matches_final = match_descriptors(desc1, desc2, best_ratio)
    result = draw_matches(img1, img2, kp1, kp2, inliers)
    os.makedirs("match/output", exist_ok=True)
    cv2.imwrite("match/output/matches.png", result)
    print("Saved to match/output/matches.png")
    if H is not None:
        print("Homography:\n", H)


if __name__ == "__main__":
    main()
