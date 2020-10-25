import numpy as np
import tensorflow as tf

import config

###################################

image_name = 'fujiyama.jpg'
image_output_name = 'fujiyama_resized.jpg'

###################################

image_path = config.IMG_INPUT_DIR + image_name

image = tf.keras.preprocessing.image.load_img(image_path)

input_arr = tf.keras.preprocessing.image.img_to_array(image)
height,width,_ = input_arr.shape

input_arr = np.array([input_arr])  # Convert single image to a batch.print(input_arr.shape)

resize_layer = tf.keras.layers.experimental.preprocessing.Resizing(height//2,width//2,interpolation='bilinear')
    
resized_image = resize_layer(input_arr)[0]

tf.keras.preprocessing.image.save_img(config.IMG_OUTPUT_DIR + image_output_name,resized_image)








