from flask import Flask, request, jsonify,render_template
#from flask_cors import CORS
import dotenv
from tone_classifier_utilites import predict_result

#load_dotenv()    


app = Flask(__name__)

@app.route('/')
def home():
     return "The test server is running"

@app.route('/predict')
def index():
     return "server Responded"

@app.route('/predict',methods=["POST"])
def predict():
     text=request.json.get('text')
     tone_classification_result=predict_result(text)
     result=jsonify(tone_classification_result)

     return result 


if __name__ == '__main__':
    app.run(debug=True)