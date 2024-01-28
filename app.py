# Import necessary libraries
from flask import Flask, render_template, jsonify, request
import subprocess

# Create a Flask app
app = Flask(__name__)

# Define the route for your web page
@app.route('/')
def index():
    return render_template('index.html')

# Define a new route to trigger your speech-to-speech script
@app.route('/run-speech-to-speech', methods=['GET','POST'])
def run_speech_to_speech():
    print("NOONEI")
    try:
        # Execute your Python voice assistant script
        result = subprocess.check_output(['python', '/mnt/c/Users/amrit/speech_to_speech_college_chat/speech_to_speech.py'], text=True, stderr=subprocess.STDOUT)
        return jsonify({'output': result.strip()})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e.output)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)