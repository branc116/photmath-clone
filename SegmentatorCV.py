import cv2
import numpy as np
class ImageDef():
  def __init__(self, im, coords):
    # "" coords -> (x, y, width, height) ""
    self.im = im
    self.coords = coords

class SegmentatorCV():
    def __init__(self, maxBound=25.0):
      self.maxBound = maxBound
    def segmentFile(self, fileName):
      im = cv2.imread(fileName)
      return self.segmentImage(im)
    def segmentImage(self, im):
      gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
      a = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
      contours, hiararhy = a[-2:]
      images = []
      for cnt in contours:
        dst = np.ones([28, 28]) + 255
        x,y,w,h = cv2.boundingRect(cnt)
        sf = max(w, h) / self.maxBound
        roi=gray[y:y+h,x:x+w]
        out = cv2.resize(roi, (int(roi.shape[1] / sf), int(roi.shape[0] / sf)), interpolation=cv2.INTER_AREA)
        l = 14 - int(out.shape[0] / 2)
        r = 14 + int(out.shape[0] / 2) + out.shape[0] % 2
        t = 14 - int(out.shape[1] / 2)
        b = 14 + int(out.shape[1] / 2) + out.shape[1] % 2
        dst[l:r, t:b] = out
        dst[:, :] = ((255 - dst)/255.0)
        images.append(ImageDef(dst, (x, y, w, h)))
      images.sort(key=lambda x: x.coords[0])
      return [image.im.reshape([1, 28, 28, 1]) for image in images if image.coords[2] < im.shape[1] * 0.7]
