import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import csv
from flask import Flask, render_template,redirect, request
import pandas as pd
app = Flask(__name__)

def text_splitter(text): #without this function vectorizer.pkl can't be loaded
  return text.split()

def vect_to_lable(v):
    global y_list
    l = []
    for i in range(len(v)):
        if v[i] == 1:
            l.append(y_list[i])
    return l

def pred(user_input):
    user_input_vect = vectorizer_saved.transform(user_input)
    pred_result = model_saved.predict(user_input_vect)
    pred_result = pred_result.toarray()
    pred_result= pred_result.tolist()
    pred_result = vect_to_lable(pred_result[0])
    return pred_result
f = open("vectorizer.pkl","rb")
vectorizer_saved = pickle.load(f)
f = open("tag_predictor_model.pkl","rb")
model_saved = pickle.load(f)

y_list = pd.read_csv("All_tags.csv")
y_list = y_list['Tags'].tolist()
#y_list[10]
#print("Hello")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    features = [features[0] + " " + features [1]]
    output = pred(features)
    '''
    global output
    output = get_news(int_features)
    output = output.upper()
    '''
    return render_template('index.html', prediction_text='The House Price for the selected options would be : {}'.format(output))
    #return redirect(url_for('showresult'))


if __name__ == "__main__":
    app.run(debug=True)