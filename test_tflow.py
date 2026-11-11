import tensorflow as tf

print("TensorFlow Version:", tf.__version__)

print("Available GPUs:", tf.config.list_physical_devices('GPU'))

print("TensorFlow is working!")