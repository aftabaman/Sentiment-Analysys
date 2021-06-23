from flask import Flask, render_template,request
from lime import lime_text
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words =set(stopwords.words('english'))
exclude_words = set(("not", "no"))
stop_words = stop_words.difference(exclude_words)
import pickle




app = Flask(__name__)


model=pickle.load(open("assets/SVM_model.sav", 'rb'))


# preprocessing the data
# opereations will be
# 1> Casing
# 2>nois removing
# 3> tokenizing
# 4>stopwords removing
# 5>text normalization(stemming and lemmatizing)

def preProcessing(text):
    # removing the urls from the text
    text = re.sub(r'https?:\/\/[A-Za-z0-9\.\/]+', '', text)

    # removing the # or @ from the tweets

    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#', '', text)

    # removing RT
    text = re.sub(r'RT : ', '', text)
    # camelCase spliting

    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', text)).split()

    print(splitted)
    temp = " ".join(splitted)
    text = temp

    # changing all words into lowercase
    text = text.lower()

    # removeing punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))

    # removing emoji
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    text = regrex_pattern.sub(r'', text)

    # removing stopwords

    text_tokens = word_tokenize(text)
    print(text_tokens)
    filtered_words = []

    for word in text_tokens:
        if word not in stop_words:
            filtered_words.append(word)

    # stemming
    # ps = PorterStemmer()
    # stemmed_words =[]
    #
    # for word in filtered_words:
    #  stemmed_words.append(ps.stem(word))

    # lemmatizing
    lemmatizer = WordNetLemmatizer()
    lemma_word = []
    for word in filtered_words:
        lemma_word.append(lemmatizer.lemmatize(word, pos='a'))

    return " ".join(lemma_word)




def predict(model,text):
    clean_text = preProcessing(text)
    result = model.predict([clean_text])
    proba_temp = model.predict_proba([clean_text]).tolist()

    proba = proba_temp[0]

    result= result.tolist()
    dic={'class_labels':['AntiVaccine','Neutral','ProVaccine'],'result':result[0],'proba':proba,'data':[],'labels':[]}

    classes=['AntiVax','Neutral','ProVax']
    explainer = lime_text.LimeTextExplainer(class_names=classes)
    explained = explainer.explain_instance(clean_text,model.predict_proba, num_features=5,labels=[0,1,2])

    for i in range(3):
        if classes[i]==result[0]:
            data_pack = explained.as_list(label=i)


    label = []
    data = []

    for i in data_pack:
        data.append(round(i[1],2))
        label.append(i[0])

    dic['data'].extend(data)
    dic['labels'].extend(label)

    print(dic['labels'])
    return dic


predict(model,"this vaccine is not good ,don't take it.")



@app.route('/')
def hello_world():

    return render_template("index.html")



@app.route('/analyze-text',methods=["POST"])

def analyze():
    if request.method=="POST":
        print("ok",request.values['text'])
        if request.values:
            text = request.values["text"] # geting the image that has been passed

            dic = predict(model, text)
            return render_template("test.html", details=dic)

    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True)