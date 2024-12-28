# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    
    # Encoding delle variabili categoriche, serve a convertire le variabili categoriche in variabili numeriche
    df = pd.get_dummies(df, drop_first=True)
    
    # Separazione delle feature (X) e della variabile target (y)
    X = df.drop('LCtype', axis=1)  # la variabile target è 'LCtype'
    y = df['LCtype']
    
    # Scaling delle feature
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y
