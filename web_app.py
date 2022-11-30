from flask import Flask, request, render_template
from eliza_punjabi import ActionEliza
from emotion_detector_utils import EmotionDetector
from sklearn.feature_extraction import DictVectorizer

import regex 
import random
import pickle
#should be added
# from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
# import torch
# import torch.nn.functional as F
# import ktrain
# from ktrain import text
# import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#delete for deployment
# import argparse
# import nltk
# # from nltk.corpus import wordnet
# from nltk import word_tokenize
# from nltk import pos_tag


#debuging in this machine only
# from datetime import datetime

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('ChattyMira.html')

@app.route('/', methods=['POST'])
def my_form_post():
    def load_pkl(fname):
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj

    input_sentence = request.form['text'] #"ਕੀ ਤੁਸੀਂ ਇੱਕ ਚੰਗੇ ਬੋਟ ਹੋ?"

    emotion_detector = EmotionDetector()

    # loading xgboost and linearSVC emotion detectors along with vectorizer for tweets
    with open('./punj_emoiton_SVC_vectorizer.pkl', 'rb') as fid:
        svc_model,vectorizer = pickle.load(fid)

    xgboost = load_pkl('./ED_punj_xgboost.pkl')
    
    # input_sentence = emotion_detector.rem_stopwords(input_sentence)
    features = emotion_detector.create_feature(input_sentence)
    features = vectorizer.transform(features)
    detected_emotion = xgboost.predict(features)[0]


    eEliza = ActionEliza()
    test_data = {}
    

    print('\n\n\n')
    ### static response from rule based
    rule_based_key_resp = eEliza.generate_final_response(input_sentence, 3, False, detected_emotion)
    keyword = rule_based_key_resp["key"] 
    rule_based_resp = rule_based_key_resp["response"]

    ### Dynamic response from rule based
    print('\n\n\nDynamic response from rule based')
    rule_based_dynamic_key_resp = eEliza.generate_final_response(input_sentence, 3, True,detected_emotion)
    dynamic_keyword = rule_based_dynamic_key_resp["key"] 
    dynamic_rule_based_resp = rule_based_dynamic_key_resp["response"] 

    final_response = (rule_based_resp , dynamic_rule_based_resp)
    # print( ' ==================== THE FINAL RESPONESE IS ==============',final_response)    



    
    return render_template("ChattyMira.html",text = final_response[0],text2 = final_response[1], text3 = detected_emotion)
    
if __name__ == "__main__":
    app.run(debug=True)   
