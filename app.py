from flask import Flask, request, render_template
import os 
import numpy as np
import h5py
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import json

# Define custom deserialization function for BatchNormalization
class CustomBatchNormalization(BatchNormalization):
    @classmethod
    def from_config(cls, config):
        if isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return super(CustomBatchNormalization, cls).from_config(config)

# Update custom objects in Keras
tf.keras.utils.get_custom_objects().update({'BatchNormalization': CustomBatchNormalization})

# Function to load the model
def load_custom_model(filepath):
    with h5py.File(filepath, 'r') as f:
        model_config = f.attrs.get('model_config')
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        model_config = json.loads(model_config)

        # Adjust axis for BatchNormalization layers
        for layer in model_config['config']['layers']:
            if layer['class_name'] == 'BatchNormalization':
                if isinstance(layer['config']['axis'], list):
                    layer['config']['axis'] = layer['config']['axis'][0]

        model = model_from_json(json.dumps(model_config), custom_objects={'CustomBatchNormalization': CustomBatchNormalization})
        
        # Load weights
        for layer in model.layers:
            if isinstance(layer, CustomBatchNormalization):
                layer.trainable = False
        model.load_weights(filepath)
    return model

app = Flask(__name__)
model = load_custom_model('model_new1.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def innerpage():
    return render_template("inner-page.html")

@app.route("/submit", methods=["POST", "GET"])
def submit():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        
        img = load_img(filepath, target_size=(224, 224))
        # Convert the image to an array and normalize it
        image_array = np.array(img) / 255.0
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        # Use the pre-trained model to make a prediction
        predictions = model.predict(image_array)
        class_labels = ['combat', 'destroyed_building', 'fire', 'humanitarian', 'vehicles']
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        print("Predicted class:", predicted_class_label)
        return render_template("portfolio-details.html", predict=predicted_class_label)

if __name__ == "__main__":
    app.run(debug=False, port=1234)

