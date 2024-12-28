# classifier.py
from src.data_loader import load_data, clean_data
from src.preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Passo 0: Caricamento  del dataset
current_dir = os.getcwd()  
local_file_path  = "\VariableStarClassification\data\PLV_LINEAR.csv"
fullDataSetPath = current_dir + local_file_path
df = load_data(fullDataSetPath)  # path corretto per il dataset

# Passo 1: Pulizia del dataset 
df = clean_data(df)

# Passo 2: Preprocessing dei dati
X, y = preprocess_data(df)

# Passo 3: Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializza il modello KNN con il parametro k (ad esempio, k=3)
for k in range (1,25):
    knn = KNeighborsClassifier(n_neighbors=k)

    # Allena il modello sui dati di addestramento
    knn.fit(X_train, y_train)

    # Fai previsioni sul set di test
    y_pred = knn.predict(X_test)

    # Calcola l'accuratezza del modello
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza del modello KNN: {accuracy * 100:.2f}%")

    #Come controllo la correttezza della previsione?
    

