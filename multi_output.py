import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from IPython.display import Image

inputs = keras.Input(shape=(28, 28, 1), name='Input')

hidden1 = layers.Conv2D(64, kernel_size=3, activation='relu', name='hidden1')(inputs)
hidden2 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', name='hidden2')(hidden1)
hidden3 = layers.Conv2D(64, kernel_size=3, strides=2, activation='relu', name='hidden3')(hidden2)
flatten = layers.Flatten()(hidden3)

age_output = layers.Dense(1, name='Age_Output')(flatten)
gender_output = layers.Dense(1, name='Gender_Output')(flatten)

model = keras.Model(inputs=inputs, outputs=[age_output, gender_output])

# 產生網絡拓撲圖
plot_model(model, to_file='Functional_API_Multi_Output_Model.png')

# 秀出網絡拓撲圖
Image('Functional_API_Multi_Output_Model.png')