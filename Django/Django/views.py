import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler  
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from django.shortcuts import render

def home(request):
    return render(request, 'index.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    if request.method == 'GET':
        if not all(request.GET.get(key) for key in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']):
            return render(request, 'predict.html', {'error_message': 'Please provide all feature values.'})
        else:
            data = pd.read_csv('../Dataset/train_card_transdata.csv')
            # split the data into features & target
            X = data.drop('fraud', axis=1)
            y = data['fraud']
    
            # Perform random over-sampling to deal with imbalanced classes
            ros = RandomOverSampler(random_state=42)
            X_res, y_res = ros.fit_resample(X, y)
    
            # first, split the data into the training & test sets
            xTrain, xTest, yTrain, yTest = train_test_split(X_res, y_res, test_size=0.33, random_state=42)
    
            # random forest model creation 
            rf_model = RandomForestClassifier() 
            rf_model.fit(xTrain, yTrain)
    
            val1 = float(request.GET['n1'])
            val2 = float(request.GET['n2'])
            val3 = float(request.GET['n3'])
            val4 = float(request.GET['n4'])
            val5 = float(request.GET['n5'])
            val6 = float(request.GET['n6'])
            val7 = float(request.GET['n7'])
    
            pred = rf_model.predict([[val1, val2, val3, val4, val5, val6, val7]])
    
            result = ""
            if pred == 1:
                result = "Legitimate transaction"
            else:
                result = "Fraudulent transaction"
                
            return render(request, 'predict.html', {'result': result})
    else:
        return render(request, 'predict.html')
