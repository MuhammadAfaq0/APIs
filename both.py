from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained models
cnn_model_path = 'model/parkinsons_model.h5'  # Update with your actual model path
cnn_model = load_model(cnn_model_path)
rnn_model_path = 'model/parkinsons_rnn_Final.h5'  # Update with your actual model path
rnn_model = load_model(rnn_model_path)

# Example questions for RNN model
questions = [
    "Have you been getting slower in your usual daily activities?",
    "Is your handwriting smaller?",
    "Is your speech slurred or softer?",
    "Do you have trouble rising from a chair?",
    "Do your lips, hands, arms and/or legs shake?",
    "Have you noticed more stiffness?",
    "Do you have trouble fastening buttons or dressing?",
    "Do you shuffle your feet and/or take smaller steps when you walk?",
    "Do your feet seem to get stuck to the floor when walking or turning?",
    "Have you or others noticed that you don't swing one arm when walking?"
]

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
        prediction = cnn_model.predict(processed_image)
        prediction = prediction[0]  # Get the first element (since batch size is 1)
        # Interpret prediction
        if prediction > 0.5:
            result = "Parkinson's Disease (PD)"
        else:
            result = "Control (No PD)"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# API endpoint for image prediction
@app.route('/api/predict', methods=['POST'])
def api_predict_parkinsons():
    try:
        # Get the file from the POST request
        file = request.files['file']
        # Save the file to disk temporarily
        temp_path = 'temp_upload.jpg'
        file.save(temp_path)
        # Preprocess the image
        processed_image = preprocess_image(temp_path)
        # Make prediction
        prediction = cnn_model.predict(processed_image)
        prediction = prediction[0]  # Get the first element (since batch size is 1)
        # Remove the temporary file
        os.remove(temp_path)
        # Interpret prediction
        if prediction > 0.5:
            result = "Parkinson's Disease (PD)"
        else:
            result = "Control (No PD)"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

# Endpoint to start the questionnaire
@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'GET':
        return render_template('questionnaire.html', questions=questions)
    elif request.method == 'POST':
        # Process form submission with user answers
        user_answers = []
        for i in range(len(questions)):
            answer = request.form.get(f'answer_{i}')  # Form input names like 'answer_0', 'answer_1', etc.
            # Convert answer to numerical value
            if answer == 'none':
                user_answers.append(0)
            elif answer == 'sometimes':
                user_answers.append(1)
            elif answer == 'severe':
                user_answers.append(2)
            else:
                return "Invalid answer submitted."

        # Calculate total score out of 30
        total_score = sum(user_answers)

        # Calculate percentage
        percentage = (total_score / 30) * 100

        # Prepare input data for prediction (assuming model requires this format)
        input_data = np.array([user_answers])  # Assuming user_answers is a list of numerical values

        # Make prediction using your model
        prediction = rnn_model.predict(input_data)
        print(percentage)

        # Determine result based on the highest prediction value
        result = "Control" if prediction[0][0] > percentage else "Parkinson's Disease"

        return render_template('result.html', result=result, percentage=percentage)
    else:
        return "Method not allowed"

# API endpoint to start the questionnaire
@app.route('/api/questionnaire', methods=['GET', 'POST'])
def apiquestionnaire():
    if request.method == 'GET':
        return jsonify({"questions": questions})
    elif request.method == 'POST':
        data = request.json
        print(data)
        user_answers = data.get('features', [])
        
        if len(user_answers) != len(questions):
            return jsonify({"error": "Invalid number of answers submitted."}), 400
        
        print(user_answers)
        # Calculate total score out of 30
        total_score = sum(user_answers)

        # Calculate percentage
        percentage = (total_score / 30) * 100

        # Prepare input data for prediction (assuming model requires this format)
        input_data = np.array([user_answers])  # Assuming user_answers is a list of numerical values

        # Make prediction using your model
        prediction = rnn_model.predict(input_data)

        # Determine result based on the highest prediction value
        result = "Control" if prediction[0][0] > percentage else "Parkinson's Disease"

        return jsonify({"result": result, "percentage": percentage})
    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
