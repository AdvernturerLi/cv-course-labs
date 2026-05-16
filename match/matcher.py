import numpy as np


def match_descriptors(desc1, desc2, ratio=0.75, cross_check=True):

    if len(desc1) == 0 or len(desc2) == 0:
        return []

    # ||a-b||^2 = 2 - 2 a·b
    sim = desc1 @ desc2.T
    dist = np.sqrt(np.maximum(2.0 - 2.0 * sim, 0.0))

    # 正向：
    idx_sorted = np.argsort(dist, axis=1)
    nn1 = idx_sorted[:, 0]
    nn2 = idx_sorted[:, 1] if dist.shape[1] >= 2 else idx_sorted[:, 0]
    rows = np.arange(len(desc1))
    d1 = dist[rows, nn1]
    d2 = dist[rows, nn2]
    # 防 0 除
    d2_safe = np.where(d2 > 1e-12, d2, 1e-12)
    keep = d1 < ratio * d2_safe
    fwd = {int(i): int(nn1[i]) for i in np.where(keep)[0]}

    if not cross_check:
        return list(fwd.items())

    # 反向
    distT = dist.T
    idx_sorted_b = np.argsort(distT, axis=1)
    nn1b = idx_sorted_b[:, 0]
    nn2b = idx_sorted_b[:, 1] if distT.shape[1] >= 2 else idx_sorted_b[:, 0]
    rows_b = np.arange(len(desc2))
    db1 = distT[rows_b, nn1b]
    db2 = distT[rows_b, nn2b]
    db2_safe = np.where(db2 > 1e-12, db2, 1e-12)
    keep_b = db1 < ratio * db2_safe
    bwd = {int(j): int(nn1b[j]) for j in np.where(keep_b)[0]}

    return [(i, j) for i, j in fwd.items() if bwd.get(j) == i]
