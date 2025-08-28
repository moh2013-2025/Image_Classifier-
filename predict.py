import json
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import warnings 
warnings.filterwarnings('ignore')
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

import glob 
from PIL import Image

from tensorflow.keras import layers


# Creat a parser
parser = argparse.ArgumentParser(description="A command-line application (Python script) that predicts the type of a given flower out of 102 different species of flowers")

# Positional arguments 
parser.add_argument("image_path", help="path to the input image folder", type=str)
parser.add_argument("Project_Image_Classifier_Project", help="path to the saved Keras model", type=str)

# Optional arguments 
parser.add_argument("-k", "--top_k", default=3, help ="top k class probabilities", type=int)
parser.add_argument("-n", "--category_names", default="./label_map.json", help="path to a JSON file mapping labels to the actual flower names", type=str)

args = parser.parse_args()   



image_size = 224
def process_image(image):
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    _image = np.expand_dims(image, axis = 0)
    predictions =  model.predict(_image)
    top_k_values, top_k_indices = tf.nn.top_k(predictions, k= top_k)
    return top_k_values.numpy(), top_k_indices.numpy(), image


if __name__ == "__main__":
    print("Start Prediction...")
    
    # Load class names from JSON file
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    # Load the saved model
    reloaded_model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={'KerasLayer': hub.KerasLayer}
    )
    
    # Make predictions
    probs, classes, _ = predict(args.image_path, reloaded_model, args.top_k)
    label_names = [class_names[str(idd)] for idd in classes]  # Use str(idd) as keys must match JSON format
    
    # Display results
    print("Probabilities:", probs)
    print("Classes:", classes)
    print("Labels:", label_names)
    
    print("End Prediction")

