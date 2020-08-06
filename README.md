# Pokerpro
This is a game of poker which uses Neural Network to guess the best hand and hence help the user make an informed decision
It uses Multi Layer Perceptron in order to predict the hand. 
The model has an accuracy of more than 90%
It uses two hidden layers containing 32 and 16 neurons respectively




DESCRIPTION OF FILES

Dataset: 
The dataset used is the from UCI Machine Learning repository and is titled Poker Hand Dataset
Link : http://archive.ics.uci.edu/ml/datasets/Poker+Hand

The dataset consists of 25000 entries in the trining file and 100000 entries in the testing file.


api.py: 
This is a flask api which enables hosting og the model and hence obtain predictions.
This api is used for hosting via heroku.

model.pk1: 
This file is created as a model from the code.
This file can be used in order to avoid retraining the data.

scaler.save: 
This file stores the configurations of the input.
Hence it is used to convert the test data into appropriate form in order to be acceptable by the model.


model_creation.py: 
Multi Layer Perceptron is used in order to create the model and train it.
MLP is used because it provides high accuracy.
This code creates a model which can be used for future predictions

run_model.py: 
This file just contains the code to run the model and predict the hand from file test1.csv
