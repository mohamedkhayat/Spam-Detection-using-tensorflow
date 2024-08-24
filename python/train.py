import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from clean import clean_text
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle

CLASSES = ['ham','spam']

#Loading clean data

data = pd.read_csv('python/data/clean_data.csv')

#dropping na values

data = data.dropna()

longest_sentence = data['Message'].apply(len).mean()
print(f"The longest sentence is:\n{longest_sentence}")

#extracting features and labels into numpy arrays

features = data['Message'].values
labels = data['Category'].values

#tokenizing sentences so that the model can train on them

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)

#padding sequences to be of consistent length so that the model can accept them as input

padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen = 64 , padding='post')

#printing padded sequences to make sure padding has been conducted correctly,each sentence should be of length 20

print(padded_sequences.shape)

#Splitting data into train and test groups

x_train,x_val,y_train,y_val = train_test_split(padded_sequences,labels,test_size=0.2,random_state=31)

#Checking shapes to make sure the split has been conducted correctly

print(f'X train shape :{x_train.shape}')
print(f'X test shape :{x_val.shape}')
print(f'y train shape :{y_train.shape}')
print(f'y test shape :{y_val.shape}')

#Specifying model structure using keras's sequential api
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(64,)),
    tf.keras.layers.Embedding(input_dim=10000,output_dim=64,input_length = 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.001))),
    tf.keras.layers.Dropout(0.3),  # Dropout layer added
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,kernel_regularizer=tf.keras.regularizers.L2(0.001))),
    tf.keras.layers.Dropout(0.3),  # Dropout layer added
    tf.keras.layers.Dense(32,activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

#printing a summary of the model to make sure all is well

model.summary()

#compiling the model while specifiying optimizer,learning rate,loss function and metrics to track while training

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

#training the model on x_train and y_train and conducting validation using x_val and y_val for 10 epochs on batches of 32 examples
#saving the accuracy loss and epochs to history for visualization later

history =model.fit(
    x_train,
    y_train,
    validation_data = (x_val,y_val),
    epochs = 10,
    batch_size =32,
    verbose = 1
)
#Final validation score

val_loss,val_accuracy = model.evaluate(x_val,y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

#plotting loss per epoch for validation and training loss

plt.plot(history.history['loss'],color='blue',label='train')
plt.plot(history.history['val_loss'],color="red",label='val')
plt.title('train/val loss')
plt.xlabel('Epoch')
plt.ylabel("loss")
plt.legend()
plt.show()

#plotting accuracy per epoch for validation and training loss

plt.plot(history.history['accuracy'],color="blue",label="train")
plt.plot(history.history['val_accuracy'],color="red",label="val")
plt.title("train/val Accuracy")
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.legend()
plt.show()

#getting predicted labels on validation set for confusion matrix

test_pred = (model.predict(x_val))
test_pred = (test_pred > 0.5).astype(int)

#plotting confusion matrix
cm = confusion_matrix(y_val, test_pred, labels=[0, 1])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

save = int(input("do you want so save the model ? yes (1) or no (0): "))
if save==1:
    version = 'v1'
    model.export("python/model"+version)

    with open('python/tokenizer.pkl','wb') as handle:
        pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
        print("tokenizer saved")
        