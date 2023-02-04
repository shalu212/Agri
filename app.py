from flask import Flask,render_template,url_for,request,redirect
import numpy as np
import pandas as pd
import joblib
import pickle

app=Flask(__name__)

model=joblib.load('decmodel.pkl')


@app.route('/')
@app.route('/main')
def main():
	return render_template('main.html')

@app.route('/predict',methods=['post'])
def predict():
	int_features=[[x for x in request.form.values()]]
	
	c=['N','P','K','temperature','humidity','ph','rainfall']
    
	fin=pd.DataFrame(int_features,columns=c)
	result=model.predict(fin)
	return render_template("main.html",prediction_text=" The Recommended Crop is : {}".format(result))
	




if __name__ =="__main__":
    app.run(host = '0.0.0.0', port=8089, debug =True)
