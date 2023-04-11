import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the vectorizer and classifier from disk
vectorizer = joblib.load('vectorizer.joblib')
clf = joblib.load('classifier.joblib')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the form
    text = request.form['text']

    # Convert the text data to numerical data using the vectorizer
    X = vectorizer.transform([text])

    # Make a prediction using the classifier
    y_pred = clf.predict(X)[0]

    # Return the predicted label as a response
    return y_pred


if __name__ == '__main__':
    app.run(debug=True)