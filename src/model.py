#Cargar librerías
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
import json
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# Leer el archivo JSON
json_file =  os.getcwd()+"\\config\\config.json"

with open(json_file, 'r') as f:
    cfg = json.load(f)

# Cargar datos
dat1 = pd.read_csv(cfg['root_folder'] + "\\data\\"+cfg['data_model'])
data1=dat1.copy()
data1 = data1.drop(['id','timestamp'], axis=1)
dat2 = pd.read_csv(cfg['root_folder'] + "\\data\\data_model2.csv")
data2=dat2.copy()
data2 = data2.drop(['id','timestamp'], axis=1)
dat3 = pd.read_csv(cfg['root_folder'] + "\\data\\data_model3.csv")
data3=dat3.copy()
data3 = data3.drop(['id','timestamp'], axis=1)


# modelos kmeans con 3 combinaciones de variables en 3 escenarios
for i in [2,3,4]:
    kmeans_model = KMeans(n_clusters=i)
    kmeans_model.fit(data1)
    kmeans_labels = kmeans_model.labels_
    dat1[f'kmeans{i}']=kmeans_model.labels_

for i in [2,3,4]:
    kmeans_model = KMeans(n_clusters=i)
    kmeans_model.fit(data2)
    kmeans_labels = kmeans_model.labels_
    dat2[f'kmeans{i}']=kmeans_model.labels_

for i in [2,3,4]:
    kmeans_model = KMeans(n_clusters=i)
    kmeans_model.fit(data3)
    kmeans_labels = kmeans_model.labels_
    dat3[f'kmeans{i}']=kmeans_model.labels_

#modelos isolation forest con 3 combinaciones de variables en 3 escenarios
for i in [0.01,0.05,0.1,0.2]:
    isolation_model = IsolationForest(contamination=i)
    isolation_model.fit(data1)
    isolation_scores = isolation_model.decision_function(data1)
    dat1[f'isolation_score{i}']=isolation_scores
    dat1[f'isolation_pred{i}']=isolation_model.predict(data1)


for i in [0.01,0.05,0.1,0.2]:
    isolation_model = IsolationForest(contamination=i)
    isolation_model.fit(data2)
    isolation_scores = isolation_model.decision_function(data2)
    dat2[f'isolation_score{i}']=isolation_scores
    dat2[f'isolation_pred{i}']=isolation_model.predict(data2)

for i in [0.01,0.05,0.1,0.2]:
    isolation_model = IsolationForest(contamination=i)
    isolation_model.fit(data3)
    isolation_scores = isolation_model.decision_function(data3)
    dat3[f'isolation_score{i}']=isolation_scores
    dat3[f'isolation_pred{i}']=isolation_model.predict(data3)

#modelo svm one class procesamiento largo un escenarios en las 3 combinaciones de variables

svm_model = OneClassSVM(nu=0.01)
svm_model.fit(data1)
svm_scores = svm_model.decision_function(data1)
dat1[f'svm_score{i}']=svm_scores
dat1[f'svm_pred{i}']=svm_model.predict(data1)

svm_model = OneClassSVM(nu=0.01)
svm_model.fit(data2)
svm_scores = svm_model.decision_function(data2)
dat2[f'svm_score{i}']=svm_scores
dat2[f'svm_pred{i}']=svm_model.predict(data2)

svm_model = OneClassSVM(nu=0.01)
svm_model.fit(data3)
svm_scores = svm_model.decision_function(data3)
dat3[f'svm_score{i}']=svm_scores
dat3[f'svm_pred{i}']=svm_model.predict(data3)



#modelo kmeans manual para detección anomalias
def GetDistanceByPoint(ClusterDataset, ClusterModel):
    ClusterDistance = pd.Series()
    for Cluster in range(0, len(ClusterDataset)):
        FirstPoint = np.array(ClusterDataset.loc[Cluster])
        SecondPoint = ClusterModel.cluster_centers_[ClusterModel.labels_[Cluster] - 1]
        ClusterDistance.at[Cluster]= np.linalg.norm(FirstPoint - SecondPoint)
    return ClusterDistance

Cluster = 10
OutliersFraction = 0.01
    
Kmeans = KMeans(n_clusters = Cluster).fit(data1)
KmeansDistance = GetDistanceByPoint(data1, Kmeans)
NumberOfOutliers = int(OutliersFraction*len(KmeansDistance))
Threshold = KmeansDistance.nlargest(NumberOfOutliers).min()
dat1['AnomalyKMeans'] = (KmeansDistance >= Threshold).astype(int)
dat1['Cluster'] = Kmeans.predict(data1)


#modelo autoencoder
#input_dim = data.shape[1]
#capa codificacion
#encoding_dim = 10

#input_layer = tf.keras.Input(shape=(input_dim,))
#max, activacion simple
#encoder = Dense(encoding_dim, activation='relu')(input_layer)
#probabilidades
#decoder = Dense(input_dim, activation='sigmoid')(encoder)

#autoencoder = Model(inputs=input_layer, outputs=decoder)
#autoencoder.compile(optimizer='adam', loss='mse')
#epochs iteraciones entrenamiento, size: muestras, 
#autoencoder.fit(data, data, epochs=10, batch_size=32, shuffle=True, validation_split=0.3)
#reconstructed = model.predict(data)
#error = np.mean(np.power(data - reconstructed, 2), axis=1)
#labels = [1 if e > np.mean(error) + np.std(error) else 0 for e in error]
#labels.sum()

#guardar los modelos con mejor tasa decision function
joblib.dump(isolation_model, cfg['root_folder'] + '/models/isolation.pkl')
joblib.dump(kmeans_model, cfg['root_folder'] + '/models/kmeans.pkl')

#guardar datos
dat1.to_csv(cfg['root_folder'] + "/data/"+'model.csv', index=False)
dat2.to_csv(cfg['root_folder'] + "/data/"+'model2.csv', index=False)
dat3.to_csv(cfg['root_folder'] + "/data/"+'model3.csv', index=False)
