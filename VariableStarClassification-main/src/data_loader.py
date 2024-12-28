# src/data_loader.py
import numpy as np
import pandas as pd

def load_data(path):

    return pd.read_csv(path, delimiter=';')

def clean_data(df):

    # Rimuove qualsiasi tipo di carattere di spazio dai nomi delle colonne
    df.columns = df.columns.str.replace(r'\s+', '', regex=True)

    # Rimuove la colonna con soli valori NaN
    df = df.drop(columns=['#'])

    # Rimuove righe con NaN
    df = df.dropna()

    # Rimuove righe con valori infiniti
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]

    # Rimuove duplicati
    df = df.drop_duplicates()

    print("\nAnteprima dei dati caricati:\n", df.head())

    return df
