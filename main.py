import pandas as pd
from pyswip import Prolog
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

from decision_tree_entropia import DecisionTreeEntropia
from decision_tree_gini import DecisionTreeGini

from sklearn.model_selection import StratifiedKFold

def create_balanced_chunks(df, target_column='target_class', n_chunks=18):

    skf = StratifiedKFold(n_splits=n_chunks, shuffle=True, random_state=42)
    chunks = []

    for _, chunk_indices in skf.split(df, df[target_column]):
        chunks.append(df.iloc[chunk_indices])

    return chunks

def train_and_predict_py(X_train, Y_train, X_test, Y_test):

    models = [
        ('Gini', DecisionTreeGini(max_depth=4, min_samples_leaf=1, min_information_gain=0.05)),
        ('Entropia', DecisionTreeEntropia(max_depth=4, min_samples_leaf=1, min_information_gain=0.05))
    ]
    
    for model_name, model in models:
        print(f"========== Train in Python with {model_name} ===========")
        
        # Allena il modello
        model.train(X_train, Y_train)
        # Predizioni
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Matrice di confusione
        cm = confusion_matrix(Y_test, test_preds)
        print(f"Confusion Matrix {model_name} in PYTHON:")
        TN, FP, FN, TP = cm.ravel()  # Esplode la matrice in 4 valori
        
        # Calcola l'accuratezza
        accuracy = (TN + TP) / (TN + FP + FN + TP)
        
        print(f"Accuracy {model_name}: {accuracy:.4f}")
        print(f"True Negatives {model_name}: {TN}")
        print(f"False Positives {model_name}: {FP}")
        print(f"False Negatives {model_name}: {FN}")
        print(f"True Positives {model_name}: {TP}")


def train_and_predict_pl(X_train, Y_train, X_test, Y_test, prolog):
    export_to_prolog(X_train, Y_train, 'train_data.pl')
    export_to_prolog(X_test, Y_test, 'test_data.pl')
    print("========== Train in Prolog ===========")
    
    run_tree_query = "run_tree."
    list(prolog.query(run_tree_query))

def export_to_prolog(X,Y,filename):
    with open(filename, 'w') as f:
        if(filename=='train_data.pl'):
            for features, label in zip(X.values, Y.values):
                feature_list = ', '.join(map(str, features))
                f.write(f"dtrain([{feature_list}], {label}).\n")
        else:
            for features, label in zip(X.values, Y.values):
                feature_list = ', '.join(map(str, features))
                f.write(f"dtest([{feature_list}], {label}).\n")

def cross_validation():
    df = pd.read_csv('pulsar_stars.csv')
    chunks = create_balanced_chunks(df)

    prolog = Prolog()
    prolog.query("set_prolog_flag(stack_limit, 3*10**9).")
    prolog.consult('decision_tree_entropia.pl')

    for idx, chunk in enumerate(chunks):
        print(f"========== ANALIZING CHUNK NUMBER {idx + 1} ===========")
        X = chunk.drop('target_class', axis=1)
        Y = chunk['target_class']

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

        train_and_predict_py(X_train, Y_train, X_test, Y_test)
        train_and_predict_pl(X_train, Y_train, X_test, Y_test, prolog)

cross_validation()