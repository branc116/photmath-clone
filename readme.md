#Photomath clone

Made something like a photomath clone using 2 neural networks. One that was trained on minst dataset, and second that was traind on 
dataset containing.

Models and training data is written in it's own section.

This reposoitory contains 2 NN w8s in h5 format and in tensorflow js model.
This repository contains web application that uses 2 NNs and tryes to evaluate the expression in browser using tensorflowJs 

## DEMO IS [HERE](https://ricko.us.to/photomath)



## Modeli

### Model 1 - minst dataset

Epoch 120/120
* 159ms/step - loss: 0.0050
* sparse_categorical_accuracy: 0.9993
* val_loss: 0.0276
* val_sparse_categorical_accuracy: 0.9926
* Model: "sequential"

|Layer (type)|                 Output Shape|              Param # |
|:------|:------:|:------:|
|conv2d (Conv2D)              |(None, 24, 24, 50)        |1300      |
|max_pooling2d (MaxPooling2D) |(None, 12, 12, 50)        |0         |
|conv2d_1 (Conv2D)            |(None, 8, 8, 50)          |62550     |
|max_pooling2d_1 (MaxPooling2 |(None, 4, 4, 50)          |0         |
|flatten (Flatten)            |(None, 800)               |0         |
|dense (Dense)                |(None, 64)                |51264     |
|dense_1 (Dense)              |(None, 10)                |650       |

* Total params: 115,764
* Trainable params: 115,764
* Non-trainable params: 0

### Model 2 - kaggle datasets download -d michelheusser/handwritten-digits-and-operators

Epoch 2000/2000
* 1s 59ms/step
* loss: 0.2989
* sparse_categorical_accuracy: 0.9085
* val_loss: 0.3040
* val_sparse_categorical_accuracy: 0.9069
* Model: "sequential_13"


|Layer (type) |                 Output Shape |              Param # |
|:--:|:---:|:---:|
| conv2d_26 (Conv2D)           | (None, 24, 24, 2)         |52        |
| max_pooling2d_26 (MaxPooling | (None, 12, 12, 2)         |0         |
| conv2d_27 (Conv2D)           | (None, 8, 8, 2)           |102       |
| max_pooling2d_27 (MaxPooling | (None, 4, 4, 2)           |0         |
| flatten_13 (Flatten)         | (None, 32)                |0         |
| dense_26 (Dense)             | (None, 16)                |528       |
| dense_27 (Dense)             | (None, 16)                |272       |

* Total params: 954
* Trainable params: 954
* Non-trainable params: 0
_________________________________
