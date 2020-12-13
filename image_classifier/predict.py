import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from argparse import ArgumentParser
from os import path
from PIL import Image
from json import load

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = ArgumentParser(description='Get the type of a flower.')
parser.add_argument('image', type=str, help="The image path")
parser.add_argument('model', type=str, help="your model path")
parser.add_argument('--top_k', help='Return the top K most likely classes:', type=int, default=5)
parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names:', type=str, default='./label_map.json')
args = parser.parse_args()

IMG_SHAPE = 224

image = args.image
model_path = args.model
top_k = args.top_k
category_names = args.category_names

def process_image(in_image):
    image = tf.convert_to_tensor(np.asarray(in_image))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SHAPE, IMG_SHAPE))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    img = Image.open(image_path)
    processed_test_image = process_image(img)
    results = model.predict(np.expand_dims(processed_test_image, axis=0))
    top_classes = (-results[0]).argsort()[:top_k]
    top_probs = results[0][top_classes]
    return top_probs, top_classes

def process_predictions(predictions):
    with open(category_names, 'r') as f:
        class_names = load(f)
        mapped_results = [(class_names[str(r + 1)], p) for r, p in zip(predictions[1], predictions[0])]
        return mapped_results
        

if not path.exists(image):
    print('Image at path "%s" does not exist' % image)
    
elif not path.exists(model_path):
    print('Model at path "%s" does not exist' % model_path)

elif __name__ == '__main__':
    keras_model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})

    predictions = predict(image, keras_model, top_k)

    mapped_predictions = process_predictions(predictions)

    for c, p in mapped_predictions:
        print ('%s: %1.3f' % (c, p))
    
