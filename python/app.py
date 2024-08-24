from flask import Flask,request,jsonify
import tensorflow as tf
import numpy as np
import pickle
from clean import clean_text

CLASSES = ['ham','spam']

app = Flask(__name__)

#loading the saved model for inference

version = 'v1'
model = tf.saved_model.load("python/model"+version)
infer = model.signatures['serving_default']

#loading pickled tokenizer for inference
with open('python/tokenizer.pkl','rb') as handle:
    tokenizer = pickle.load(handle)

#pre processing function to transform the data to the same form used in training
def preprocess(text,tokenizer,max_len=64):

    #clean the text same way as during trainging
    cleantext = [clean_text(sentence) for sentence in text]
    
    #transform the data to sequences during tokenizer
    sequences = tokenizer.texts_to_sequences(cleantext)
    
    #pad sequences to same size as during training
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen=max_len,padding='post')

    return padded_sequences

@app.route('/predict',methods=['POST'])
def predict():
    try:
        #read json sent over from the java app
        data = request.json 

        #checks if sentence is in data, if not return error code
        if 'sentence' not in data:
            
            return jsonify({"error":'missing sentence field'}),500
            
        #extract the sentence to use for prediction
        sentence = data.get('sentence')
        
        try:
            #apply the pre processing
            preprocessed_text = preprocess([sentence],tokenizer)

        except Exception as e:
            #return error if any that occured during preprocessing
            return jsonify({"error":'Error pre processing sentence'}),500
            
        try:
            #transform the array to a float32 tensor and making a prediction
            input_tensor = tf.constant(preprocessed_text,dtype=tf.float32)
            predictions = infer(input_tensor)['output_0'].numpy() 

            #map the prediction to one of the classes 
            prediction = CLASSES[round(predictions[0][0])]
            
            #return the prediction as a json back to the java application
            return jsonify({'result': prediction})
            
        except Exception as e:
            #return error during inference
            return jsonify("error",f"Error during inference : {str(e)}"),500
            
    except Exception as e:
        #return if any other unspecified error occured
        return jsonify({"error":f"An unexpected error occured: {str(e)}"}),500
        
#running the flask application
if __name__ == '__main__':
    app.run(debug=True)