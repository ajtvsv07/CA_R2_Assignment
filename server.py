from flask import Flask, render_template
from flask.json import jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train():
    return jsonify(
        result= "training code is yet to be written"
    )

@app.route("/infer")
def infer():
    # output = predict()
    return jsonify(
        result= "Inference code is yet to be written"
    )
