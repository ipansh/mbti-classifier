from flask import Flask, request
import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
mbti_model = joblib.load("mbti_model.pkl")

application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello World!"

@application.route('/mbti_detector', methods = ['GET', 'POST'])
def mbti_detector():
    message = request.args.get("message")
    vect_message = vectorizer.transform([message])
    result = mbti_model.predict(vect_message)[0]
    return result

if __name__  == '__main__':
    application.run(port=5000, debug=True)