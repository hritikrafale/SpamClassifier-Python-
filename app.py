from flask import Flask, request, redirect, url_for
from flask.templating import render_template
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from math import log, sqrt
import pandas as pd
import numpy as np
import re
from joblib import dump, load 
import nltk
nltk.download('punkt')

app = Flask(__name__)

@app.route("/spam_classifier",methods=["GET","POST"])
def spam_classifier():
    if request.method == "POST":
        prob_spam, prob_ham, sum_tf_idf_spam, sum_tf_idf_ham, spam_words, ham_words,prob_spam_mail,prob_ham_mail = getData()
        message = request.form["message"]
        processed_message = process_message(message)
        result = classify(processed_message , 'tf-idf' , prob_spam, prob_ham, sum_tf_idf_spam, sum_tf_idf_ham, spam_words, ham_words,prob_spam_mail,prob_ham_mail)
        return redirect(url_for("result",result=result, mssg=message))
    else:
        return render_template("index.html")

@app.route("/<result>/<mssg>")
def result(result,mssg):
    return render_template("result.html",content=result,mssg=mssg)

def getData():
    prob_spam = load("./Joblib/prob_spam_tf_idf.joblib")
    prob_ham = load('./Joblib/prob_ham_tf_idf.joblib')
    sum_tf_idf_spam = load('./Joblib/sum_tf_idf_spam.joblib')
    sum_tf_idf_ham = load('./Joblib/sum_tf_idf_ham.joblib')
    spam_words = load('./Joblib/spam_words.joblib')
    ham_words = load('./Joblib/ham_words.joblib')
    prob_spam_mail = load('./Joblib/prob_spam_mail.joblib')
    prob_ham_mail = load('./Joblib/prob_ham_mail.joblib')
    return prob_spam, prob_ham, sum_tf_idf_spam, sum_tf_idf_ham, spam_words, ham_words,prob_spam_mail,prob_ham_mail

def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 2):
    #converting all upper case characters to lower case
    if lower_case:
        message = message.lower()
    
    # word tokenize converts sentence into an array of words
    words = word_tokenize(message)

    #all the words of length less than or equal to 2 are discarded
    words = [w for w in words if len(w) > 2]
    return words

    #gram is used to slide the window of size 2 
    #since gram is set to 2 we will have pair of two words 
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w

    # stop words are words like 'a' 'the' 'in' etc so below we are removing all the stop words     
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]

    # stemming is the process of reducing words to there base form like reducing words like eating, eats, eaten to eat    
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]   
    
    return words

def classify(processed_message , method, prob_spam, prob_ham, sum_tf_idf_spam, sum_tf_idf_ham, spam_words, ham_words,prob_spam_mail,prob_ham_mail):
        pSpam, pHam = 0, 0
        for word in processed_message:                
            if word in prob_spam:
                pSpam += log(prob_spam[word])
            else:
                if method == 'tf-idf':
                    pSpam -= log(sum_tf_idf_spam + len(list(prob_spam.keys())))
                else: 
                    pSpam -= log(spam_words + len(list(prob_spam.keys())))
            if word in prob_ham:
                pHam += log(prob_ham[word])
            else:
                if method == 'tf-idf':
                    pHam -= log(sum_tf_idf_ham + len(list(prob_ham.keys()))) 
                else:
                    pHam -= log(ham_words + len(list(prob_ham.keys())))
            pSpam += log(prob_spam_mail)
            pHam += log(prob_ham_mail)
        return pSpam >= pHam


if __name__ == "__main__":
    app.run(debug=True)