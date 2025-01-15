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

        stats.append({
            'phase': 'Python',
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

                # Converti i valori della matrice di confusione in interi
                TN, FP, FN, TP = map(int, parts[1:])

                # Aggiungi ai dati statistici
                stats.append({
                    'phase': 'Prolog',
                    'model': model_name,
                    'accuracy': accuracy,
                    'confusion_matrix': {
                        'TN': TN,
                        'FP': FP,
                        'FN': FN,
                        'TP': TP
                    },
                })

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

def average_validation():
    df = pd.read_csv('pulsar_stars.csv')
    chunks = create_balanced_chunks(df)

    all_stats = []

    prolog = Prolog()
    prolog.query("set_prolog_flag(stack_limit, 3*10**9).")

    for idx, chunk in enumerate(tqdm(chunks, desc="Cross-validation progress")):
        X = chunk.drop('target_class', axis=1)
        Y = chunk['target_class']

        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=42)

        stats = train_and_test(X_train, Y_train, X_test, Y_test, prolog)
        all_stats.extend(stats)

    aa = list(prolog.query('classify_example([24.484375, 41.16544067, 4.347698018, 18.75114241, 109.7892977, 82.21609152, 0.010425467, -1.259046483],PredictedClass).'))

    print(aa)

    return all_stats


def average_stats(all_stats):
    model_stats = {}

    for stat in all_stats:
        model_name = stat['model']
        if model_name not in model_stats:
            model_stats[model_name] = {
                'accuracy': {'Python': [], 'Prolog': []},
                'TN': {'Python': [], 'Prolog': []},
                'FP': {'Python': [], 'Prolog': []},
                'FN': {'Python': [], 'Prolog': []},
                'TP': {'Python': [], 'Prolog': []}
            }

        phase = stat['phase']
        model_stats[model_name]['accuracy'][phase].append(stat['accuracy'])
        model_stats[model_name]['TN'][phase].append(stat['confusion_matrix']['TN'])
        model_stats[model_name]['FP'][phase].append(stat['confusion_matrix']['FP'])
        model_stats[model_name]['FN'][phase].append(stat['confusion_matrix']['FN'])
        model_stats[model_name]['TP'][phase].append(stat['confusion_matrix']['TP'])

    for model_name, values in model_stats.items():
        # Stampa per Python
        avg_accuracy_python = sum(values['accuracy']['Python']) / len(values['accuracy']['Python'])
        avg_TN_python = sum(values['TN']['Python']) / len(values['TN']['Python'])
        avg_FP_python = sum(values['FP']['Python']) / len(values['FP']['Python'])
        avg_FN_python = sum(values['FN']['Python']) / len(values['FN']['Python'])
        avg_TP_python = sum(values['TP']['Python']) / len(values['TP']['Python'])

        print(f"Model: {model_name}")
        print(f"  Python - Average Accuracy: {avg_accuracy_python:.4f}")
        print(f"  Python - Average Confusion Matrix:")
        print(
            f"    TN: {avg_TN_python:.2f}, FP: {avg_FP_python:.2f}, FN: {avg_FN_python:.2f}, TP: {avg_TP_python:.2f}")

        # Stampa per Prolog
        avg_accuracy_prolog = sum(values['accuracy']['Prolog']) / len(values['accuracy']['Prolog']) if \
        values['accuracy']['Prolog'] else 0
        avg_TN_prolog = sum(values['TN']['Prolog']) / len(values['TN']['Prolog']) if values['TN']['Prolog'] else 0
        avg_FP_prolog = sum(values['FP']['Prolog']) / len(values['FP']['Prolog']) if values['FP']['Prolog'] else 0
        avg_FN_prolog = sum(values['FN']['Prolog']) / len(values['FN']['Prolog']) if values['FN']['Prolog'] else 0
        avg_TP_prolog = sum(values['TP']['Prolog']) / len(values['TP']['Prolog']) if values['TP']['Prolog'] else 0

        print(f"  Prolog - Average Accuracy: {avg_accuracy_prolog:.4f}")
        print(f"  Prolog - Average Confusion Matrix:")
        print(
            f"    TN: {avg_TN_prolog:.2f}, FP: {avg_FP_prolog:.2f}, FN: {avg_FN_prolog:.2f}, TP: {avg_TP_prolog:.2f}")


stats = average_validation()
average_stats(stats)
