import dash
from dash import dcc
from dash import html
import dash_split_pane
import plotly.express as px
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install dash_daq
import dash_daq as daq
import shap
from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, fbeta_score, confusion_matrix, auc
import plotly.figure_factory as ff
import joblib
from joblib import load
import base64
from PIL import Image
from shap.plots._force_matplotlib import draw_additive_plot
from dash import dash_table
from flask import Flask, render_template, jsonify, request
import json
import requests
from waitress import serve
import pickle



cheminFichierJoblib = './fichierJoblib/'

predictionStreamlit = joblib.load(cheminFichierJoblib + 'predictionStreamlit.joblib')
predictionProbaStreamlit = joblib.load(cheminFichierJoblib + 'predictionProbaStreamlit.joblib')

X_testID2 = joblib.load(cheminFichierJoblib + 'X_testID2.joblib')
dataframeInfoXTest2 = joblib.load(cheminFichierJoblib + 'dataframeInfoXTest2.joblib')
listeIndexSort = joblib.load(cheminFichierJoblib + 'listeIndexSort.joblib')
dfIdClientIndex = joblib.load(cheminFichierJoblib + 'dfIdClientIndex.joblib')

app = Flask(__name__)
app.title = "Projet 7"  # Assigning title to be displayed on tab

@app.route('/')
def listeID():
    data = []  # On initialise une liste vide
    for id in listeIndexSort:
        idClient = id
        data.append(idClient)

    return jsonify({'status': 'ok', 'Liste ID Client a notre disposition': data})


@app.route('/infoClient', methods=['GET'])
def informationClient():
    if 'id' in request.args:
        user_id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    nombreDEnfant = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == user_id]['CNT_CHILDREN'].values
    revenu = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == user_id]['AMT_INCOME_TOTAL'].values
    montantDuCredit = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == user_id]['AMT_CREDIT'].values
    annuiteDePret = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == user_id]['AMT_ANNUITY'].values
    dureeRemboursement = montantDuCredit / annuiteDePret

    infoClient = [
        {
            'Nombre d\'enfant': int(nombreDEnfant[0]),
            'Revenu du client': int(revenu[0]),
            'Montant du credit': int(montantDuCredit[0]),
            'Annuite de pret': int(annuiteDePret[0]),
            'Duree du remboursement': float(dureeRemboursement[0])
        }
    ]

    return jsonify(infoClient)


@app.route('/predictionClient', methods=['GET'])
def predictionClient():
    if 'id' in request.args:
        user_id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    target = dataframeInfoXTest2[dataframeInfoXTest2['SK_ID_CURR'] == user_id]['TARGET'].values
    scorePred = X_testID2[X_testID2['ID'] == user_id]['Score prediction'].values
    scoreProba = X_testID2[X_testID2['ID'] == user_id]['Score prediction proba'].values
    if scorePred == 0:
        pred_text = 'Negative'
    else:
        pred_text = 'Positive'

    predictionClient = [
        {'Le client recoit son credit': pred_text,
         'Target': int(target[0]),
         'Score prediction': float(scorePred[0]),
         'Score prediction proba': float(scoreProba[0])
         }
    ]

    return jsonify(predictionClient)


@app.route('/predictionModeleClient', methods=['GET'])
def predict():
    if 'id' in request.args:
        user_id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    index = dfIdClientIndex[dfIdClientIndex['idClient'] == user_id].index
    index = int(index[0])

    # pkl_file = open('lgbmHPSeuil.pkl', 'rb')
    # lgbmHPSeuil = pickle.load(pkl_file)
    # y_predLgbmHPSeuil = lgbmHPSeuil.predict_proba(X_test)[::,1]
    # prediction = lgbmHPSeuil.predict(X_test)
    # predictionProba = lgbmHPSeuil.predict_proba(X_test)[::,1]

    prediction = predictionStreamlit
    predictionProba = predictionProbaStreamlit

    y_predSeuil = np.zeros(prediction.shape)
    y_predSeuil[prediction > 0.1] = 1

    if y_predSeuil[index] == 0:
        pred_text = 'Negative'
    else:
        pred_text = 'Positive'

    # round the predict proba value and set to new variable
    confidence = round(predictionProba[index], 3)
    output = {'prediction': pred_text, 'confidence': confidence}

    return output


if __name__ == '__main__':
    #app.run(debug=True)
    serve(app, host='0.0.0.0')
