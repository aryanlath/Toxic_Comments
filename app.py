import flask
from flask import Flask
from flask import request
from config import *
import torch
from model import *
import numpy as np
import gc
import os
import re
import sys
from flask import Flask, jsonify, request, render_template, flash
from wtforms import Form, TextField, TextAreaField, validators, SubmitField

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)

print(current_dir, parent_dir)
sys.path.append(parent_dir)

app = Flask(__name__)

app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

class ReusableForm(Form):
    comment = TextAreaField('Comment:', validators=[validators.required()])

def comment_prediction(comment):
    toxic_preds = []
    comment_tokenized = CONFIG['tokenizer'].encode_plus(
                            comment,
                            truncation=True,
                            add_special_tokens=True,
                            max_length=CONFIG['max_length'],
                            padding='max_length'
                        )
    comment_in = torch.tensor([comment_tokenized['input_ids']], dtype=torch.long)
    comment_am = torch.tensor([comment_tokenized['attention_mask']], dtype=torch.long)
    comment_tti = torch.tensor(0, dtype=torch.long)
    model = ToxicModel(CONFIG['model_name'])
    model.to(CONFIG['device'])
    model.load_state_dict(torch.load('Epoch-2.bin', map_location = torch.device('cpu')))
    outputs = model(comment_in.to(CONFIG['device'], dtype = torch.long) , comment_am.to(CONFIG['device'], dtype = torch.long), comment_tti.to(CONFIG['device'], dtype = torch.long))
    outputs = np.array(torch.sigmoid(outputs).cpu().detach().numpy())
    toxic_preds.append(outputs) 
    gc.collect()
    return toxic_preds

def output(toxic_preds):
    out = []
    for a in toxic_preds:
        out.append(a)
    out = np.array(out)
    out = out.reshape(6)
    topic = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for a in range(6):
        print(topic[a])
        print(out[a])
    return out
        
    

@app.route('/', methods=['GET', 'POST'])
def deploy():
    out = []
    form = ReusableForm(request.form)
    print(form.errors)
    if request.method == 'POST':
            print("Form:")
            print(form)
            #name = request.form['name']
            comment = request.form['comment']
            #print(f"Name:{name}")
            print(f"Comment:{comment}")
            
            if form.validate():
                toxic_preds = comment_prediction(comment)
                out = output(toxic_preds)
            else:
                flash('Error: All the form fields are required. ')
    return render_template('index.html', form=form, prediction = out)



if __name__ == '__main__':
    app.run(debug = True)
