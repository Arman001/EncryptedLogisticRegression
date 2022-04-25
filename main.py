# Cread By Muhammad Saad #
# On 24/04/22           #
# 4:00 Pm               #
#.......................#

from logistic_regression import logistic
import numpy as np
from Data_Preprocessing import load_data
import pickle

with open('./log_weights.pkl', 'rb') as file:
            loaded_weights = pickle.load(file)
# log_obj = logistic(784, 0)
log_obj = logistic(loaded_weights, 1)
def train(X, Y):
    acc = 0
    lr = 0.001
    loss = 0.0
    print("---------------Training is Starting---------------------")
    for epoch in range(10):
        print(f"Epoch: {epoch+1}")
        for i in range(len(X)):
            out, loss = log_obj.training(X[i], Y[i], lr, 0)
            if((out>0.5 and Y[i]==1) or (out<0.5 and Y[i]==0)):
                acc = acc + 1
            
            if(i!=0 and (i+1)%10000==0):
                print(f"|{i+1} items done|Accuracy:{int((acc/10000)*100)}%|Loss:{loss/10000}|")
                acc = 0
                loss = 0.0
    print("---------------Training has Ended---------------------")
    

def test(X, Y):
    acc = 0
    print("----------------Testing is Starting---------------------")

    for i in range(len(X)):
        out, loss = log_obj.training(X[i], Y[i], 0, 1)
        if((out>0.5 and Y[i]==1) or (out<0.5 and Y[i]==0)):
            acc = acc + 1
    
    print(f"|{i+1} items done|Accuracy:{int((acc/len(X))*100)}%|Loss{loss/len(X)}|")
    print("---------------Testing has Ended----------------------")
    print("------------------------------------------------------")

def Check():
    with open('./unseenX.pkl', 'rb') as file:
        X = pickle.load(file)
    with open('./unseenY.pkl', 'rb') as file:
        Y = pickle.load(file)
    test_X = X[:100]
    test_Y = Y[:100]
    test(test_X, test_Y)


    

def main():
    print("Loading Data")
    class_names, X, Y = load_data()
    X = X/255
    train_X = X[:60000]
    train_Y = Y[:60000]
    test_X = X[60000:80000]
    test_Y = Y[60000:80000]
    unseen_X = X[80000:]
    unseen_Y = Y[80000:]
    print(f"Classes are {class_names}")
    train(train_X,train_Y)
    test(test_X, test_Y)
    print("Storing the data")
    with open('unseenX.pkl', 'wb') as file:
        pickle.dump(unseen_X, file)
    with open('unseenY.pkl', 'wb') as file:
        pickle.dump(unseen_Y, file)
    print("------------------------------------------------------")
    print("Storing trained parameters")
    print("------------------------------------------------------")
    with open('log_weights.pkl', 'wb') as file:
        pickle.dump(log_obj.weights, file)

if __name__==Check():
    Check()