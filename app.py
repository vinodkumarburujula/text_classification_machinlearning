
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained ML pipeline
model = joblib.load("news_classifier.joblib")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("news_text", "").strip()

        if input_text:
            prediction = model.predict([input_text])[0]

    return render_template(
        "index.html",
        prediction=prediction,
        input_text=input_text
    )

if __name__ == "__main__":
    app.run(debug=True)
