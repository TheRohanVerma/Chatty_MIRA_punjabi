from flask import Flask, request, render_template
from eliza_punjabi import ActionEliza
from emotion_detector_utils import EmotionDetector
from sklearn.feature_extraction import DictVectorizer
import regex 
import random
import pickle

app = Flask(__name__)
@app.route('/')

def my_form():
    return render_template('ChattyMira.html') # this is the main web page that has interface for chatty mira

@app.route('/', methods=['POST'])
def my_form_post():
    def load_pkl(fname):
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj

    # loads the vectorizer and linear svc model for predictions
    svc_model,vectorizer = load_pkl('punj_emoiton_SVC_vectorizer.pkl')

    
    input_sentence = request.form['text'] #"ਕੀ ਤੁਸੀਂ ਇੱਕ ਚੰਗੇ ਬੋਟ ਹੋ?"

    emotion_detector = EmotionDetector()

    features = emotion_detector.create_feature(input_sentence)
    features = vectorizer.transform(features)
    detected_emotion = svc_model.predict(features)[0]

    # now we have the detected emotion and will use it to generate an empathetic or neutral emotion

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
    
    # after we have the final emotion, we render the home page again with the responses
    return render_template("ChattyMira.html",text = final_response[0],text2 = final_response[1], text3 = detected_emotion)
    
if __name__ == "__main__":
    app.run(debug=True)   
