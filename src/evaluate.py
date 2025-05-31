import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
import pickle 
import yaml 
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report 
import mlflow 
from mlflow.models import infer_signature 
import os 

from sklearn.model_selection import train_test_split , GridSearchCV
from urllib.parse import urlparse

## MLFLOW tracking
os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/SadabAli/MachinrLerningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="SadabAli"
os.environ['MLFLOW_TRACKING_PASSWORD']="95437498b5a15638d8aa35540cc26470566bc98d"

## loading the parameters from params.yaml 

params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X=data.drop(columns='Outcome')
    y=data['Outcome'] 

    ## MLFLOW TRACKING
    mlflow.set_tracking_uri('https://dagshub.com/SadabAli/MachinrLerningPipeline.mlflow')

    ## Load th emodel from the disk 
    model = pickle.load(open(model_path,'rb'))

    prediction = model.predict(X)
    accuracy_score=accuracy_score(y,prediction)

    ## Log matrix to mlflow
    mlflow.log_metric('accuracy',accuracy_score)
    print(f"model accuracy{accuracy_score}")

if __name__ == '__main__':
    evaluate(params['data'],params['model'])