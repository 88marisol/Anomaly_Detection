from flask import Flask, request, json
import boto3
import joblib
BUCKET_NAME = 'https://modeloapi.s3.amazonaws.com/'
MODEL_FILE_NAME = 'modelo_final.pkl'
app = Flask(__name__)
S3 = boto3.client('s3', region_name='us-east-1')

def load_model(key):    
    # Load model from S3 bucket
    response = S3.get_object(Bucket=BUCKET_NAME, Key=key)
# Load pickle model
    model_str = response['Body'].read()     
    model = joblib.load(model_str) 
    return model

@app.route('/anomaly', methods=['POST'])
def anomaly():    
    # Parse request body for model input 
    body_dict = request.get_json(silent=True)    
    data = body_dict['data']     
    
    # Load model
    model = load_model(MODEL_FILE_NAME)
# Make prediction 
    prediction = model.predict(data).tolist()
# Respond with prediction result
    result = {'prediction': prediction}    
   
    return json.dumps(result)
if __name__ == '__main__':    
    # listen on all IPs 
    app.run(host='0.0.0.0')
    
