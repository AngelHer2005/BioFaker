# app.py
from flask import Flask, jsonify, render_template
from modelo import BioFakerIA

app = Flask(__name__)
biofaker = BioFakerIA()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generar")
def generar():
    datos = biofaker.generate()
    return jsonify(datos)

if __name__ == "__main__":
    app.run(debug=True)
