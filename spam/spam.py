from flask import Flask, redirect, request, render_template
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

swords = stopwords.words('english')
ps = PorterStemmer()

def clean_text(sent):
    text = [word for word in word_tokenize(sent) 
            if word not in string.punctuation]
    text = [ps.stem(word.lower()) for word in text 
            if word.lower() not in swords]
    return text
    
    
app = Flask(__name__)

@app.route('/')
def home():
   return render_template('read.html')

@app.route('/predict', methods = ['POST'])
def predict():
    import joblib  
    classifier = joblib.load('classifier.model')
    tfidf = joblib.load('tfidf.model')
    msg = str(request.form['msg'])
    new = np.array([msg])
    y_pred = classifier.predict(tfidf.transform(new))
    return render_template('predict.html', result = y_pred[0])

if __name__ == '__main__':
   app.run(debug = True)
   
