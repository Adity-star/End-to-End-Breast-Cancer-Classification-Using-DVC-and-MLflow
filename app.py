from flask import Flask, requests, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from BCClassifier.pipeline.prediction import PredictionPipeline
from BCClassifier.utils.common import decodeImage

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
       self.filename = "inputImage.jpg"
       self.classifier = PredictionPipeline(self.filename)

@app.route("/",methods=['GET'])
@cross_origin()

def home():
    return render_template('index.html')


@app.route("/train",methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system("Python main.py")
    return "Training completed successfully"

@app.route("/predict", methods=['POST'])
@cross_origin()
def predicRoute():
    image = requests.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0",port=8080)
    