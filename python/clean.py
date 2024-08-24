import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text,stop_words):
    words = text.split()
    clean_words = [word for word in words if word not in stop_words]
    return ' '.join(clean_words)

def remove_html(text):
    pattern = r'<.*?>'
    text = re.sub(pattern,' ',text)
    return text

def remove_double_whitespaces(text):
    text = ' '.join(text.split())
    return text
    
def remove_punct(text):
    pattern = r'[^\w\s]'
    text = re.sub(pattern,' ',text)
    return text


def clean_text(text,stop_words=stop_words):
    text = text.lower()
    text = remove_punct(text)
    text = remove_html(text)
    text = remove_stopwords(text,stop_words)
    text = remove_double_whitespaces(text)
    return text


if __name__=='__main__':
    #load csv file with the data
    spam_df = pd.read_csv("python/data/spam.csv")
    #encode ham to 0 and spam to 1 for binary classification
    spam_df['Category'] = spam_df['Category'].map({'ham' : 0,'spam' : 1})
    #clean data
    spam_df['Message'] = spam_df['Message'].apply(clean_text)
    #save clean data to csv 
    spam_df.to_csv('data/clean_data.csv',index=False)

