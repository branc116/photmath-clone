from PIL import Image, ImageFilter, ImageDraw, ImageOps
import numpy as np
from math import floor

class Segmentation:

  def __init__(self, margine = 0.05, max_norm = 50):
    self.margine = margine
    self.max_norm = max_norm
  def segmentFile(self, fileName):
    img = Image.open(fileName)
    return self.segmentImage(img)
  def segmentImage(self, img):
    map = np.zeros((img.size))
    mm = []
    connections = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    def bfs(start):
      stack = [start]
      minn = start
      maxx = start
      i = 0
      while (len(stack) != 0):
        i = i + 1
        cur = stack.pop()
        map[cur] = 1
        minn = (min(minn[0], cur[0]), min(minn[1], cur[1]))
        maxx = (max(maxx[0], cur[0]), max(maxx[1], cur[1]))
        for a in connections:
          newxy = (cur[0] + a[0], cur[1] + a[1])
          if (newxy[0] > 0 and newxy[0] < img.size[0] and newxy[1] > 0 and newxy[1] < img.size[1] and map[newxy] == 0):
            pix = img.getpixel(newxy)
            map[newxy] = 1
            norm = np.linalg.norm(pix)
            if (norm < self.max_norm):
              stack.append(newxy)
      return [(minn[0] - 1, minn[1] - 1), (maxx[0] + 1, maxx[1] + 1)]
    def modify(i):
      slack = [30, 30]
      szx, szy = (i[1][0] - i[0][0], i[1][1] - i[0][1])
      if (szx > szy):
        slack[0] = floor(szx * self.margine)
        slack[1] = floor(slack[0] * szx / szy)
      else:
        slack[1] = floor(szy * self.margine)
        slack[0] = floor(slack[0] * szy / szx)  
      return ImageOps.invert(img.crop((i[0][0] - slack[0], i[0][1] - slack[1], i[1][0] + slack[0], i[1][1] + slack[1])))

    for i in range(img.size[0]):
      for j in range(img.size[1]):
        pix = img.getpixel((i, j))
        norm = np.linalg.norm(pix)
        if (map[i, j] == 0 and norm < self.max_norm):
          mm.append(bfs((i, j)))
        map[i, j] = 1
    ii = [modify(mm[i]) for i in range(len(mm))]
    return [(np.array(ii[i].resize((28, 28))).reshape(1, 28, 28, 3)[:,:,:, 0].reshape(1, 28, 28, 1) / 255.) for i in range(len(ii))]
