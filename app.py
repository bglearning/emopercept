import json

from flask import Flask, render_template, request
from joblib import load

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def emopercept():
    if 'text' in request.form.keys():
        text = request.form['text']
        prediction = str(model.predict([text])[0])
        emotion = inv_emotion_map[prediction]
    else:
        text = ''
        emotion = ''
    return render_template('index.html', text=text, emotion=emotion)


if __name__ == '__main__':
    model = load('out/final_model.joblib')
    with open('out/inv_emotion_map.json') as f:
        inv_emotion_map = json.load(f)
    app.run(debug=True)
