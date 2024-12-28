# cross_validation.py
from sklearn.model_selection import cross_val_score

def perform_k_fold_cross_validation(model, X, y, k=5):

    """
    Esegue la k-fold cross-validation per il modello dato.
    
    Parameters:
    - model: Il classificatore addestrato
    - X: Dati di input
    - y: Etichette di output
    - k: Numero di suddivisioni per la cross-validation
    
    Returns:
    - scores: I punteggi ottenuti per ciascuna delle k iterazioni
    """

    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')

    print(f"\nK-fold Cross-Validation Scores: {scores}")
    print(f"Mean Accuracy: {scores.mean()}")

    return scores
