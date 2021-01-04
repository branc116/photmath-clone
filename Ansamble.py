import numpy as np


class ModelDef():
  def __init__(self, model, mapFunction):
    self.model = model
    self.map = mapFunction

class Ansamble():
  def __init__(self, models, classNames):
    self.models = models
    self.classes = 0
    for modelDef in self.models:
      self.classes = max(self.classes, modelDef.model.output_shape[1])
    self.classNames = classNames
  def predict(self, imgs):
    return np.array([self._predict(img) for img in imgs])
  def _predict(self, img):
    predictions = np.array([self.map(m.model.predict(img)[0], m.map) for m in self.models])
    sr, sc = predictions.shape
    res = np.ones([sc])
    for i in range(sr):
      res = np.multiply(res, predictions[i])
    return res / np.linalg.norm(res)
  def toString(self, imgs):
    retStr = ""
    for pred in self.predict(imgs):
      retStr += self.classNames[np.argmax(pred)]
    return retStr
  def map(self, y, mapFunc):
    y_exp = np.exp(y)
    y_normed = y_exp / np.linalg.norm(y_exp)
    res = np.ones([self.classes])
    res[:len(y)] = y
    mapped = np.exp(mapFunc(y_normed))
    return mapped
