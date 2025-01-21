import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from decision_tree_entropia import DecisionTreeEntropia
from decision_tree_gini import DecisionTreeGini


def average_stats(all_stats, matrix):
    model_stats = {}

    for stat in all_stats:
        model_name = stat['model']
        if model_name not in model_stats:
            model_stats[model_name] = {
                'accuracy': {'Python': [], 'Prolog': []},
                'precision': {'Python': [], 'Prolog': []},
                'recall': {'Python': [], 'Prolog': []},
                'f1_score': {'Python': [], 'Prolog': []},
                'error': {'Python': [], 'Prolog': []},
                'TN': {'Python': [], 'Prolog': []},
                'FP': {'Python': [], 'Prolog': []},
                'FN': {'Python': [], 'Prolog': []},
                'TP': {'Python': [], 'Prolog': []}
            }

        phase = stat['phase']
        model_stats[model_name]['accuracy'][phase].append(stat['accuracy'])
        model_stats[model_name]['precision'][phase].append(stat['precision'])
        model_stats[model_name]['recall'][phase].append(stat['recall'])
        model_stats[model_name]['f1_score'][phase].append(stat['f1_score'])
        model_stats[model_name]['error'][phase].append(stat['error'])
        model_stats[model_name]['TN'][phase].append(stat['confusion_matrix']['TN'])
        model_stats[model_name]['FP'][phase].append(stat['confusion_matrix']['FP'])
        model_stats[model_name]['FN'][phase].append(stat['confusion_matrix']['FN'])
        model_stats[model_name]['TP'][phase].append(stat['confusion_matrix']['TP'])

    for model_name, values in model_stats.items():
        # Calcola le medie per Python
        avg_accuracy_python = sum(values['accuracy']['Python']) / len(values['accuracy']['Python'])
        avg_precision_python = sum(values['precision']['Python']) / len(values['precision']['Python'])
        avg_recall_python = sum(values['recall']['Python']) / len(values['recall']['Python'])
        avg_f1_python = sum(values['f1_score']['Python']) / len(values['f1_score']['Python'])
        avg_error_python = sum(values['error']['Python']) / len(values['error']['Python'])
        avg_TN_python = sum(values['TN']['Python'])
        avg_FP_python = sum(values['FP']['Python'])
        avg_FN_python = sum(values['FN']['Python'])
        avg_TP_python = sum(values['TP']['Python'])

        matrix.append({
            'phase': 'Python',
            'model': model_name,
            'accuracy': avg_accuracy_python,
            'precision': avg_precision_python,
            'recall': avg_recall_python,
            'f1_score': avg_f1_python,
            'error': avg_error_python,
            'confusion_matrix': {
                'TN': avg_TN_python,
                'FP': avg_FP_python,
                'FN': avg_FN_python,
                'TP': avg_TP_python
            },
        })

        # Calcola le medie per Prolog
        avg_accuracy_prolog = sum(values['accuracy']['Prolog']) / len(values['accuracy']['Prolog']) if values['accuracy']['Prolog'] else 0
        avg_precision_prolog = sum(values['precision']['Prolog']) / len(values['precision']['Prolog']) if values['precision']['Prolog'] else 0
        avg_recall_prolog = sum(values['recall']['Prolog']) / len(values['recall']['Prolog']) if values['recall']['Prolog'] else 0
        avg_f1_prolog = sum(values['f1_score']['Prolog']) / len(values['f1_score']['Prolog']) if values['f1_score']['Prolog'] else 0
        avg_error_prolog = sum(values['error']['Prolog']) / len(values['error']['Prolog']) if values['error']['Prolog'] else 0
        avg_TN_prolog = sum(values['TN']['Prolog'])
        avg_FP_prolog = sum(values['FP']['Prolog'])
        avg_FN_prolog = sum(values['FN']['Prolog'])
        avg_TP_prolog = sum(values['TP']['Prolog'])

        matrix.append({
            'phase': 'Prolog',
            'model': model_name,
            'accuracy': avg_accuracy_prolog,
            'precision': avg_precision_prolog,
            'recall': avg_recall_prolog,
            'f1_score': avg_f1_prolog,
            'error': avg_error_prolog,
            'confusion_matrix': {
                'TN': avg_TN_prolog,
                'FP': avg_FP_prolog,
                'FN': avg_FN_prolog,
                'TP': avg_TP_prolog
            },
        })

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

def create_balanced_chunks(df, target_column='target_class', n_chunks=18):

    skf = StratifiedKFold(n_splits=n_chunks, shuffle=True, random_state=42)
    chunks = []

    for _, chunk_indices in skf.split(df, df[target_column]):
        chunks.append(df.iloc[chunk_indices])

    return chunks


def train_and_test(X_train, Y_train, X_test, Y_test, prolog):
    stats = []

    models = [
        ('Gini', DecisionTreeGini(max_depth=4, min_samples_leaf=1, min_information_gain=0.05), 'decision_tree_gini.pl'),
        ('Entropia', DecisionTreeEntropia(max_depth=4, min_samples_leaf=1, min_information_gain=0.05),
         'decision_tree_entropia.pl')
    ]

    for model_name, model, prolog_file in models:
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
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        error = 1 - accuracy

        stats.append({
            'phase': 'Python',
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'error': error,
            'confusion_matrix': {
                'TN': TN,
                'FP': FP,
                'FN': FN,
                'TP': TP
            },
        })

        # ========= TRAIN E TEST IN PROLOG =========
        # Consulta il file Prolog corretto
        prolog.consult(prolog_file)

        # Esporta i dati in Prolog
        export_to_prolog(X_train, Y_train, 'train_data.pl')
        export_to_prolog(X_test, Y_test, 'test_data.pl')

        # Esegui la query per il modello
        run_tree_query = "run_tree(Result)."
        for solution in prolog.query(run_tree_query):
            result = solution['Result']  # Estratto direttamente da Prolog

            # Parsing manuale del risultato
            if isinstance(result, str):
                # Rimuovi 'result(', 'accuracy(', 'confusion_matrix(' e le parentesi finali
                result = result.replace('result(', '').replace('accuracy(', '').replace('confusion_matrix(',
                                                                                        '').rstrip('))')

                # Divide la stringa in parti
                parts = result.split(', ')

                # Rimuovi eventuali caratteri indesiderati come ')'
                parts = [part.strip(')') for part in parts]

                # Converti la prima parte in float (accuratezza)
                accuracy = float(parts[0])
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                error = 1 - accuracy

                # Converti i valori della matrice di confusione in interi
                TN, FP, FN, TP = map(int, parts[1:])

                # Aggiungi ai dati statistici
                stats.append({
                    'phase': 'Prolog',
                    'model': model_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'error': error,
                    'confusion_matrix': {
                        'TN': TN,
                        'FP': FP,
                        'FN': FN,
                        'TP': TP
                    },
                })

    return stats

def validation(prolog):
    df = pd.read_csv('pulsar_stars.csv')
    chunks = create_balanced_chunks(df)

    all_stats = []
    for idx, chunk in enumerate(tqdm(chunks, desc="Validation progress")):
        X = chunk.drop('target_class', axis=1)
        Y = chunk['target_class']

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

        stats = train_and_test(X_train, Y_train, X_test, Y_test, prolog)
        all_stats.extend(stats)

    return all_stats

def classify_data(data, criteria, prolog):
    
    prolog_file = "decision_tree_gini.pl" if criteria == "Gini" else "decision_tree_entropia.pl"
    prolog.consult(prolog_file)

    params_str = ", ".join(map(str, data))
    query = f"classify_example([{params_str}], PredictedClass)."
    result = list(prolog.query(query))
    if result:
        predicted_class = result[0]['PredictedClass']
        return predicted_class
    else:
        return "Unknown"