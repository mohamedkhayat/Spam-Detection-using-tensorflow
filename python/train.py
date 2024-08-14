import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from clean import clean_text
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASSES = ['ham','spam']

#Loading clean data

data = pd.read_csv('python/data/clean_data.csv')

#dropping na values

data = data.dropna()

#extracting features and labels into numpy arrays

features = data['Message'].values
labels = data['Category'].values

#tokenizing sentences so that the model can train on them

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000,oov_token='<OOV>')
tokenizer.fit_on_texts(features)
sequences = tokenizer.texts_to_sequences(features)

#padding sequences to be of consistent length so that the model can accept them as input

padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences,maxlen = 20 , padding='post')

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
    tf.keras.Input(shape=(20,)),
    tf.keras.layers.Embedding(input_dim=10000,output_dim=64,input_length = 20),
    tf.keras.layers.LSTM(64,return_sequences=True,kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    tf.keras.layers.LSTM(32,kernel_regularizer=tf.keras.regularizers.L2(0.001)),
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

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train/val loss')
plt.xlabel('Epoch')
plt.ylabel("loss")
plt.legend(["train,val"])
plt.show()

#plotting accuracy per epoch for validation and training loss

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("train/val Accuracy")
plt.xlabel('Epoch')
plt.xlabel('Accuracy')
plt.legend(['train,val'])
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

test_sentences = [
    "Hi, just checking in to see how you're doing. Let's catch up soon!",
    "Congratulations! You've won a $1000 gift card. Click here to claim your prize now!"
]

text_sentences = [clean_text(sentence) for sentence in test_sentences]
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=20, padding='post')

predictions = model.predict(test_padded_sequences)
for sentence, prediction in zip(test_sentences, predictions):
    print(f"Sentence: {sentence}")
    print(f"Prediction: {CLASSES[round(prediction[0])]}")