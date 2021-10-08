from flask import Flask, request, render_template
import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
mbti_model = joblib.load("mbti_model.pkl")

application = Flask(__name__)

@application.route("/")
def home_page():       
    return render_template('page.html')

@application.route("/sub", methods = ['POST'])
def submit():
    if request.method == 'POST':
        result = request.form["userpost"]
        result2 = vectorizer.transform([result])
        result3 = mbti_model.predict(result2)[0]
    return render_template('sub.html', name = result3)

if __name__  == '__main__':
    application.run(port=5000, debug=True)