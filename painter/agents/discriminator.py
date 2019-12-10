import keras
import keras.backend as K
import numpy as np

def discriminator(input_shape= (64,64,3)):
  model = keras.models.Sequential()
  #downscale
  model.add(keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.LeakyReLU(alpha=0.2))
  model.add(keras.layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
  model.add(keras.layers.Dropout(0.4))
  model.add(keras.layers.LeakyReLU(alpha=0.2))
  
  #classifieur
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation="relu"))
  model.add(keras.layers.Dense(1, activation="sigmoid"))
  #model.add(keras.layers.Dense(2, activation="softmax"))
  opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer=opt,metrics=['accuracy'])
  
  return model
