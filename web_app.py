from flask import Flask, request, render_template
from eliza_punjabi import ActionEliza
import regex 
import random

#should be added
# from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
# import torch
# import torch.nn.functional as F
# import ktrain
# from ktrain import text
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#delete for deployment
# import argparse
import nltk
# from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag


#debuging in this machine only
# from datetime import datetime

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('ChattyMira.html')

@app.route('/', methods=['POST'])
def my_form_post():
    input_sentence = request.form['text'] #"ਕੀ ਤੁਸੀਂ ਇੱਕ ਚੰਗੇ ਬੋਟ ਹੋ?"
    eEliza = ActionEliza()

    # processed_text = eEliza.sum(text)

    test_data = {}

    print('\n\n\n')
    ### static response from rule based
    # print('\n\n\nStatic response from rule based')
    # start = datetime.now()

    rule_based_key_resp = eEliza.generate_final_response(input_sentence, 3, False)
    # statistics['calculation_time_sum']['static rulebased'] += (datetime.now()-start).microseconds
    keyword = rule_based_key_resp["key"] 
    rule_based_resp = rule_based_key_resp["response"]

    # test_data[input_sentence]["static rulebased"]["output"] = rule_based_resp
    # print(test_data)

    ### Dynamic response from rule based
    print('\n\n\nDynamic response from rule based')
    # start = datetime.now()
    rule_based_dynamic_key_resp = eEliza.generate_final_response(input_sentence, 3, True)
    # statistics['calculation_time_sum']['dynamic rulebased'] += (datetime.now()-start).microseconds
    dynamic_keyword = rule_based_dynamic_key_resp["key"] 
    dynamic_rule_based_resp = rule_based_dynamic_key_resp["response"] 


    dataframe = (pd.DataFrame.from_dict(test_data)).to_string()
    # dataframe.to_csv('11_20_out.csv', index=False)
    final_response = (rule_based_resp , dynamic_rule_based_resp)
    # print( ' ==================== THE FINAL RESPONESE IS ==============',final_response)
    return render_template("ChattyMira.html",text = final_response[0],text2 = final_response[1])
    
if __name__ == "__main__":
    app.run(debug=True)   
