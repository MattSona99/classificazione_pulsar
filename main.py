import pandas as pd
from pyswip import Prolog
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from decision_tree_entropia import DecisionTreeEntropia
from decision_tree_gini import DecisionTreeGini

from sklearn.model_selection import StratifiedKFold

def create_balanced_chunks(df, target_column='target_class', n_chunks=18):

    skf = StratifiedKFold(n_splits=n_chunks, shuffle=True, random_state=42)
    chunks = []

    for _, chunk_indices in skf.split(df, df[target_column]):
        chunks.append(df.iloc[chunk_indices])

    return chunks

def train_and_test(X_train, Y_train, X_test, Y_test, prolog):

    stats = []

    models = [
        ('Gini', DecisionTreeGini(max_depth=4, min_samples_leaf=1, min_information_gain=0.05)),
        ('Entropia', DecisionTreeEntropia(max_depth=4, min_samples_leaf=1, min_information_gain=0.05))
    ]
    
    for model_name, model in models:

        # ========= TRAIN E TEST IN PYTHON =========   
        # Allena il modello
        model.train(X_train, Y_train)
        # Predizioni
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        # Matrice di confusione
        cm = confusion_matrix(Y_test, test_preds)
        TN, FP, FN, TP = cm.ravel()  # Esplode la matrice in 4 valori
        
        # Calcola l'accuratezza
        accuracy = (TN + TP) / (TN + FP + FN + TP)

        stats.append({
            'phase:': 'Python',
            'model': model_name,
            'accuracy': accuracy,
            'confusion_matrix': {
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TP': TP
            },
        })

        # ========= TRAIN E TEST IN PROLOG =========
        if(model_name=='Entropia'):
            export_to_prolog(X_train, Y_train, 'train_data.pl')
            export_to_prolog(X_test, Y_test, 'test_data.pl')
            run_tree_query = "run_tree."
            list(prolog.query(run_tree_query))

    return stats

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

    all_stats = []

    prolog = Prolog()
    prolog.query("set_prolog_flag(stack_limit, 3*10**9).")
    prolog.consult('decision_tree_entropia.pl')

    for idx, chunk in enumerate(tqdm(chunks, desc="Cross-validation progress")):
        X = chunk.drop('target_class', axis=1)
        Y = chunk['target_class']

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

        stats = train_and_test(X_train, Y_train, X_test, Y_test, prolog)
        all_stats.extend(stats)

    return all_stats

def average_stats(all_stats):
    model_stats = {}

    for stat in all_stats:
        model_name = stat['model']
        if model_name not in model_stats:
            model_stats[model_name] = {
                'accuracy': [],
                'TN': [],
                'FP': [],
                'FN': [],
                'TP': []
            }
        model_stats[model_name]['accuracy'].append(stat['accuracy'])
        model_stats[model_name]['TN'].append(stat['confusion_matrix']['TN'])
        model_stats[model_name]['FP'].append(stat['confusion_matrix']['FP'])
        model_stats[model_name]['FN'].append(stat['confusion_matrix']['FN'])
        model_stats[model_name]['TP'].append(stat['confusion_matrix']['TP'])

    for model_name, values in model_stats.items():
        avg_accuracy = sum(values['accuracy']) / len(values['accuracy'])
        avg_TN = sum(values['TN']) / len(values['TN'])
        avg_FP = sum(values['FP']) / len(values['FP'])
        avg_FN = sum(values['FN']) / len(values['FN'])
        avg_TP = sum(values['TP']) / len(values['TP'])

        print(f"Model: {model_name}")
        print(f"  Average Accuracy: {avg_accuracy:.4f}")
        print(f"  Average Confusion Matrix:")
        print(f"    TN: {avg_TN:.2f}, FP: {avg_FP:.2f}, FN: {avg_FN:.2f}, TP: {avg_TP:.2f}")

stats = cross_validation()
average_stats(stats)
