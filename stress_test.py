import time
import math
import cv2
import numpy as np

def stress(n=5000000):
    s = 0.0
    for i in range(n):
        s += math.sin(i) * math.cos(i)
    return s

start = time.time()
stress()
end = time.time()

print("Zeit:", round(end - start, 3), "s")

img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

start = time.time()
for _ in range(3000):
    cv2.resize(img, (640, 360))
print("Resize Zeit:", round(time.time()-start, 3), "s")

a = np.random.randn(3000, 3000)

start = time.time()
np.linalg.svd(a)
print("SVD Zeit:", round(time.time()-start, 3), "s")