from flask import Flask, request, render_template, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your RNN model
model = load_model('./model/parkinsons_rnn_Final.h5')

# Example questions
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
        prediction = model.predict(input_data)
        print(percentage)

        # Determine result based on the highest prediction value
        result = "Control" if prediction[0][0] > percentage else "Parkinson's Disease"

        return render_template('result.html', result=result, percentage=percentage)
    else:
        return "Method not allowed"


# Endpoint api to start the questionnaire
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
        prediction = model.predict(input_data)

        # Determine result based on the highest prediction value
        result = "Control" if prediction[0][0] > percentage else "Parkinson's Disease"

        return jsonify({"result": result, "percentage": percentage})
    else:
        return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5001,debug=True)
