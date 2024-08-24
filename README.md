### **Overview**
The **Spam Detection App** is a practical project designed to apply TensorFlow and NLP techniques for classifying spam messages. This project builds on concepts from the *Machine Learning Specialization* and the book *Natural Language Processing in Action* to create, train, and deploy an effective spam detection model.

### **Key Features**

- **Spam Detection Model:** Utilizes TensorFlow and LSTM networks to develop and train a model for spam classification.
- **Data Processing:** Implements text preprocessing steps, including tokenization, padding and stop-word removal.
- **Flask API:** Deploys the trained model using Flask for real-time spam classification.
- **Java Frontend:** Features a Java-based frontend that interacts with the Flask API to handle user inputs and display classification results.

### **Technologies Used**

- **TensorFlow:** For developing and training a RNN model for NLP.
- **Python & Flask:** For deploying the model as an API.
- **Java:** For creating the frontend application.
- **NLP Techniques:** Tokenization, stop-word removal and text preprocessing.

### **Dataset**

The model is trained on the [Email Spam Detection Dataset](https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification) from Kaggle. This dataset provides a collection of emails labeled as spam or not spam, used for training and evaluating the spam detection model.

### **Learning Objectives**

- **NLP Model Development:** Gain hands-on experience with TensorFlow to develop and train NLP models for text classification.
- **Text Preprocessing:** Apply various NLP techniques to prepare and clean textual data.
- **API Deployment:** Learn to deploy machine learning models using Flask as an API.
- **Integration Skills:** Connect a Java frontend with a Flask API to deliver a practical application.

### **Project Impact**

This project provides practical experience in developing and deploying NLP models with TensorFlow, focusing on applying machine learning techniques to real-world problems.

### **Model Limitations**

The model demonstrates good performance on low-quality, low-effort spam. However, it struggles with high-quality spam due to the training dataset’s focus on less sophisticated spam examples. This limitation highlights the need for additional training data with varied spam qualities to improve the model's robustness.

### **Video Demo**


https://github.com/user-attachments/assets/e046f6c5-fdc4-4470-8773-6f221093ceea


### **Performance Metrics**
![acc_v1](https://github.com/user-attachments/assets/dbbaa2c0-6d63-4623-b64e-aa835fd81915)


![loss_v1](https://github.com/user-attachments/assets/c07d6042-7488-4821-9a05-f14683acd4c3)


![cm_v1](https://github.com/user-attachments/assets/d3f93c0a-17f1-4226-8a4e-7975f91d16fb)
