from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model_path = 'model/parkinsons_model.h5'  # Update with your actual model path
model = load_model(model_path)

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Adjust size if necessary
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize pixel values
    return x

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_parkinsons():
    try:
        # Ensure the uploads directory exists
        os.makedirs('static/uploads', exist_ok=True)

        # Get the file from the POST request
        file = request.files['file']

        # Save the file to disk
        img_path = 'static/uploads/' + file.filename
        file.save(img_path)

        # Preprocess the image
        processed_image = preprocess_image(img_path)

        # Make prediction
        prediction = model.predict(processed_image)
        prediction = prediction[0]  # Get the first element (since batch size is 1)

        # Interpret prediction
        if prediction > 0.5:
            result = "Parkinson's Disease (PD)"
        else:
            result = "Control (No PD)"

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000
    ,debug=True)
