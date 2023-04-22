from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from tensorflow import keras
from PIL import Image
import numpy as np
import io
# Creating the flask app
app = Flask(__name__)
# defining the list
class_labels = ['Bull', 'Chicken', 'Elephant', 'Fox']
#Loading the model
model = load_model('Cap_Mod.h5')
#Defining a route for the app
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Getting the image file from the form data
    image_file = request.files['image-file'].read()

    #loading the image and preprocessing it
    image = Image.open(io.BytesIO(image_file))
    image = image.resize((224, 224))
    image = np.asarray(image) / 255.0
    image = np.expand_dims(image, axis=0)

    #making a prediction
    prediction = model.predict(image)[0]
    label = np.argmax(prediction)
    confidence = prediction[label]
    predicted_label = class_labels[label]
    
    #returning the prediction as JSON
    return jsonify({
        'label': int(label),
        'confidence': float(confidence),
        'predicted_label' : predicted_label
    })

#Running the app
if __name__ == '__main__':
    app.run(debug=True)