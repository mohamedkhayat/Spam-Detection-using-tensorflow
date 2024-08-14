from flask import Flask,request,jsonify
import tensorflow as tf
import numpy as np
import pickle
from clean import clean_text

CLASSES = ['ham','spam']

app = Flask(__name__)

version = 'v1'
model = tf.saved_model.load("python/model"+version)
infer = model.signatures['serving_default']

with open('python/tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)

def preprocess(text,tokenizer,max_len=64):
    text = [clean_text(text)]
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=max_len,padding='post')
    return padded_sequences

@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.json 
        
        if 'sentence' not in data:
            return jsonify({"error":'missing sentence field'}),500
    
        sentence = data.get('sentence')
        print(sentence)
        try:
            preprocessed_text = preprocess(sentence,tokenizer)
            print(preprocessed_text)
        except Exception as e:

            return jsonify({"error":'Error pre processing sentence'}),500
            
        try:
            input_tensor = tf.constant(preprocessed_text,dtype=tf.float32)
            predictions = infer(input_tensor)['output_0'].numpy() 
            prediction = CLASSES[round(predictions[0][0])]
            return jsonify({'result': prediction})
            
        except Exception as e:
            
            return jsonify("error",f"Error during inference : {str(e)}"),500
            
    except Exception as e:
        
        return jsonify({"error":f"An unexpected error occured: {str(e)}"}),500
        
if __name__ == '__main__':
    app.run(debug=True)