
import flask
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
import pandas as pd, os
import pickle
from sklearn.externals import joblib
# Load the model

app = flask.Flask(__name__)
app.config["DEBUG"]=True

@app.route('/',methods=['GET'])
def api_predict():
    # Get the data from the POST request.
	data = request.args['predict']    # Make prediction using model loaded from disk as per the data.
	i=0
	k=0
	l=[]
	n=np.empty(shape=(1,10))
	while(i<len(data)):
		if(data[i]=='|'):
			str2=data[k:i]
			l.append(float(str2))
			k=i+1
		i+=1  
	n=np.array(l).reshape(1,10)
	scaler = StandardScaler()
	scaler = joblib.load("scaler.save") 
	test_data=pd.DataFrame(n)
	test_data.columns=["Suit1",  "Rank1",  "Suit2",  "Rank2",  "Suit3",  "Rank3",  "Suit4",  "Rank4",  "Suit5",  "Rank5"]
	test_x = np.array(test_data)
	test_set = np.empty(test_x.shape, dtype = float)
	data_set=scaler.transform(test_set)
	for index in range(len(test_x)):
		test_set[index] = test_x[index].astype(float)
	data_test = scaler.transform(test_set)
	model = pickle.load(open('model.pk1','rb'))
	c=model.predict(data_test)  
	return (str(c[0]))
if __name__ =='__main__':
	app.run()
