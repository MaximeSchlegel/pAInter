import keras
import keras.backend as K


def generator(input_shape= (128,128,3)):
  model = keras.models.Sequential()
  model.add(keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same',input_shape = input_shape))
  model.add(keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
  model.add(keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same'))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(200, activation="relu"))
  model.add(keras.layers.Dropout(0.5))
  model.add(keras.layers.Dense(10, activation="relu"))
  opt = keras.optimizers.Adam(lr=0.001, beta_1=0.5)
  model.compile(loss="binary_crossentropy", optimizer=opt)
  
  return model

"""G = generator()
G.summary()"""
