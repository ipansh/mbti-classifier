from flask import Flask, request
import joblib

vectorizer = joblib.load("/Users/ilya/Desktop/Computer-Science/Github/mbti-classifier/tfidf_vectorizer.pkl")
mbti_model = joblib.load("/Users/ilya/Desktop/Computer-Science/Github/mbti-classifier/mbti_model.pkl")

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def spamorham():
    message = request.args.get("message")
    vect_message = vectorizer.transform([message])
    result = mbti_model.predict(vect_message)[0]
    return result

if __name__  == '__main__':
    app.run()   
