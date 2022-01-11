import cv2 as cv
import numpy as np
from util import *
import matplotlib.pyplot as plt

# img_path = './fisherman.jpg'
img_path = './lena.jpg'

img  = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
img = torch.from_numpy(img).float()
fits = [(thresh, cal_fitness(img, thresh)) for thresh in range(256)]
# show the change of fitness
plt.plot([i[0] for i in fits], [i[1]for i in fits])
plt.xlabel('grayscale_thresh')
plt.ylabel('fitness')
plt.show()
best_fit = sorted(fits, key=lambda x:x[1])[-1]
print(f'best_fit is {best_fit}')
mask = img > best_fit[0]
img = 255 * mask.int().numpy().astype(np.uint8)
b = cv.imshow('hey', img)
cv.waitKey()