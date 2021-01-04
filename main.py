import tensorflow as tf
import numpy as np
from Ansamble import Ansamble, ModelDef
from Commander import Commander
from Segmentation import Segmentation
from SegmentatorCV import SegmentatorCV
classes = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '(', ')', '/', '*', '+', '-' ]
def model1swaps(x):
  res = np.ones(16) / 2.0
  res[:10] = x * 1.2
  return res
def model2swaps(x):
  res = np.zeros(16) - 1
  res[:10] = x[4:14]
  res[10:12] = x[15:17]
  res[12:17] = x[:4]
  #res[12:14] *= 0.7
  return res

def main():
  model1 = ModelDef(tf.keras.models.load_model("models/minstM.h5"), model1swaps)
  model2 = ModelDef(tf.keras.models.load_model("models/model2.h5"), model2swaps)
  ansamble = Ansamble([model1, model2], classes)
  segmentator = Segmentation(max_norm=254)
  segmentatorcv = SegmentatorCV()
  commander = Commander(ansamble, segmentator)
  #print(commander.eval("2p2.bmp"))
  print(commander.eval("txt2.bmp"))
  #print(commander.eval("complex_expression.bmp"))
  #print(commander.eval("extreme2.bmp"))
  #segmentatorcv.writeDigetsToFolder("extreme2.bmp", "ims")
  commander.writeDigetsToFolder("txt2.bmp", "ims")
if __name__ == "__main__":
    main()