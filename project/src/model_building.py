import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, concatenate

#https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
class Model_Building:

    def USEmodel(self):

        #constructing model with three inputs
        title_input = keras.Input(shape=(768,))
        description_input = keras.Input(shape=(768,))
        domain_input = keras.Input(shape=(768,))


        # the first branch operates on the title input
        x = Dense(512, activation="relu")(title_input)
        x = Dense(256, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Model(inputs=title_input, outputs=x)


        # the second branch opreates on the description
        y = Dense(512, activation="relu")(description_input)
        y = Dense(256, activation="relu")(y)
        y = Dense(128, activation="relu")(y)
        y = Model(inputs=description_input, outputs=y)

        # combine the output of the two branches
        combined = concatenate([x.output, y.output, domain_input])

        # apply a FC layer and then a regression prediction on the
        # combined outputs

        z = Dense(512, activation="relu")(combined)
        z = Dense(128, activation="relu")(z)
        z = Dense(32, activation="relu")(z)

        z = Dense(5, activation="softmax")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x.input, y.input,domain_input], outputs=z)
        print(model.summary())
        tf.keras.utils.plot_model(model, to_file='project/output/model_plots/USEmodel_plot.png', show_shapes=True, show_layer_names=True)
        return model