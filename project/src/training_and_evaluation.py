from keras.utils import to_categorical
from tensorflow.python.keras.optimizer_v2.adam import Adam
from project.utils.plotting import *
import tensorflow as tf
print(tf.config.list_physical_devices())

def train(model_name,model,title,description,domain,job_zone,epochs,learning_rate):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['categorical_crossentropy','accuracy'])


    with tf.device('/device:GPU:0'):
        history = model.fit(
            [title,description,domain], job_zone,
            epochs=epochs,
            steps_per_epoch=32,
            validation_split=0.15
        )
    model.save_weights('project/models/model_{}.h5'.format(model_name), overwrite=True)
    plot_training_val_curves(history,model_name)