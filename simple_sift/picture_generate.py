import numpy as np
import cv2

np.random.seed(42)

img = np.zeros((256,256), dtype=np.uint8)

# 画几个圆
for (x,y,r) in [(64,64,20),(180,80,15),(100,180,25),(200,200,10)]:
    cv2.circle(img,(x,y),r,200,-1)

# 画矩形
cv2.rectangle(img,(30,150),(90,220),150,-1)

# 十字线
cv2.line(img,(0,128),(255,128),180,3)
cv2.line(img,(128,0),(128,255),180,3)

# 添加高斯噪声
noise = np.random.normal(0,15,(256,256))
img = np.clip(img.astype(np.float32)+noise,0,255).astype(np.uint8)

cv2.imwrite("./input/test_image.png",img)

print("test image generated")