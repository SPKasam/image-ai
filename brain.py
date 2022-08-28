import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2L,
    decode_predictions,
)

image = tf.keras.preprocessing.image.load_img("./godzilla.jpg")
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = tf.image.resize(input_arr, (480, 480))
input_arr = np.array([input_arr])
model = EfficientNetV2L()
predictions = decode_predictions(model.predict(input_arr))

for _, label, prob in predictions[0]:
    print(f"{label}: {prob*100}")