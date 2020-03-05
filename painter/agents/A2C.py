import keras

from painter.agents.Agent import Agent


########################################################################################################################
class A2C(Agent):

    def __int__(self):
        pass


########################################################################################################################
def generator(input_shape=(64, 64, 3),
              filter_number=64,
              latent_dim=64,
              num_hidden=3,
              size_hidden=256,
              opt="adam"):
    ## 2 imputs the images
    input_target = keras.layers.Input(input_shape)
    input_current = keras.layers.Input(input_shape)

    ## Network to process the images: extract the fetures
    net_imgs_process = keras.models.Sequential()
    net_imgs_process.add(
        keras.layers.Conv2D(filter_number, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape))
    net_imgs_process.add(keras.layers.Conv2D(filter_number, (3, 3), strides=(2, 2), padding='same'))
    net_imgs_process.add(keras.layers.Conv2D(filter_number, (3, 3), strides=(2, 2), padding='same'))
    net_imgs_process.add(keras.layers.Conv2D(filter_number, (3, 3), strides=(2, 2), padding='same'))
    net_imgs_process.add(keras.layers.Flatten())
    net_imgs_process.add(keras.layers.Dense(latent_dim))

    latent_target = net_imgs_process(input_target)
    latent_current = net_imgs_process(input_current)
    latent = keras.layers.concatenate([latent_target, latent_current])

    ## Networks to compute the action to do from the fetures
    net_policy = keras.models.Sequential()
    for _ in range(num_hidden):
        net_policy.add(keras.layers.Dense(size_hidden))
        net_policy.add(keras.layers.ReLU())
    net_policy.add(keras.layers.Dense(11))

    action = net_policy(latent)

    model = keras.Model(inputs=[input_target, input_current], outputs=action)

    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    return model


model = generator()
print(model.summary())