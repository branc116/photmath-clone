from Segmentation import Segmentation
from PIL import Image

class Commander():
  def __init__(self, model, segmentator):
    self.model = model
    self.segmentator = segmentator
  def getExpression(self, imageFileName):
    self.seg = self.segmentator.segmentFile(imageFileName)
    s = self.model.toString(self.seg)
    return s
  def eval(self, imageFileName):
    expression = self.getExpression(imageFileName)
    return expression, eval(expression)
  def writeDigetsToFolder(self, fileName, directoryToSave):
    imgs = self.segmentator.segmentFile(fileName)
    for i in range(len(imgs)):
      img = Image.fromarray(imgs[i].reshape(28, 28) * 255.0).convert("RGB")
      img.save(directoryToSave + "/" + str(i) + ".jpeg")