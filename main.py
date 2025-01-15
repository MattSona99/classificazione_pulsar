import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from pyswip import Prolog
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from decision_tree_entropia import DecisionTreeEntropia
from decision_tree_gini import DecisionTreeGini

from sklearn.model_selection import StratifiedKFold

matrix = []
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
        avg_TN_python = int(sum(values['TN']['Python']) / len(values['TN']['Python']))
        avg_FP_python = int(sum(values['FP']['Python']) / len(values['FP']['Python']))
        avg_FN_python = int(sum(values['FN']['Python']) / len(values['FN']['Python']))
        avg_TP_python = int(sum(values['TP']['Python']) / len(values['TP']['Python']))

        matrix.append({
                    'phase': 'Python',
                    'model': model_name,
                    'accuracy': avg_accuracy_python,
                    'confusion_matrix': {
                        'TN': avg_TN_python,
                        'FP': avg_FP_python,
                        'FN': avg_FN_python,
                        'TP': avg_TP_python
                    },
                })

        # Stampa per Prolog
        avg_accuracy_prolog = sum(values['accuracy']['Prolog']) / len(values['accuracy']['Prolog']) if \
        values['accuracy']['Prolog'] else 0
        avg_TN_prolog = sum(values['TN']['Prolog'])
        avg_FP_prolog = sum(values['FP']['Prolog'])
        avg_FN_prolog = sum(values['FN']['Prolog'])
        avg_TP_prolog = sum(values['TP']['Prolog'])

        matrix.append({
            'phase': 'Prolog',
            'model': model_name,
            'accuracy': avg_accuracy_prolog,
            'confusion_matrix': {
                'TN': avg_TN_prolog,
                'FP': avg_FP_prolog,
                'FN': avg_FN_prolog,
                'TP': avg_TP_prolog
            },
        })
stats = average_validation()
average_stats(stats)

# Inizializza Prolog
prolog = Prolog()

# Nomi dei parametri
parameter_names = [
    "Mean of the integrated profile",
    "Standard deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve",
]

# Funzione per il primo tasto
def open_input_screen():
    input_window = tk.Toplevel(root)
    input_window.title("Inserimento Dati")

    tk.Label(input_window, text="Inserisci i parametri:", font=("Helvetica", 14)).grid(row=0, column=0, columnspan=2, pady=10)

    # Caselle di input per i parametri
    inputs = []
    for i, name in enumerate(parameter_names):
        tk.Label(input_window, text=name + ":").grid(row=i + 1, column=0, padx=10, pady=5, sticky="e")
        entry = tk.Entry(input_window)
        entry.grid(row=i + 1, column=1, padx=10, pady=5, sticky="w")
        inputs.append(entry)

    # Menu a tendina per selezionare il criterio
    tk.Label(input_window, text="Seleziona Criterio:").grid(row=len(parameter_names) + 1, column=0, padx=10, pady=10, sticky="e")
    criteria = ttk.Combobox(input_window, values=["Gini", "Entropia"], state="readonly")
    criteria.grid(row=len(parameter_names) + 1, column=1, padx=10, pady=10, sticky="w")

    # Label per mostrare il risultato
    result_label = tk.Label(input_window, text="", fg="blue", font=("Helvetica", 12))
    result_label.grid(row=len(parameter_names) + 3, column=0, columnspan=2, pady=10)

    # Funzione per eseguire la query
    def execute_query():
        selected_criteria = criteria.get()
        if not selected_criteria:
            messagebox.showerror("Errore", "Seleziona un criterio!")
            return

        # Leggi i parametri
        params = [entry.get() for entry in inputs]
        if not all(params):
            messagebox.showerror("Errore", "Inserisci tutti i parametri!")
            return

        # Costruisci la query
        params_str = ", ".join(params)
        query = f"classify_example([{params_str}], PredictedClass)."

        # Determina il file Prolog da consultare
        prolog_file = "decision_tree_gini.pl" if selected_criteria == "Gini" else "decision_tree_entropia.pl"

        try:
            # Consulta il file Prolog
            prolog.consult(prolog_file)

            # Esegui la query
            result = list(prolog.query(query))
            if result:
                predicted_class = result[0]['PredictedClass']
                result_label.config(text=f"Classe Predetta: {predicted_class}")
            else:
                result_label.config(text="Nessuna classe predetta trovata.")
        except Exception as e:
            result_label.config(text=f"Errore durante l'interazione con Prolog:\n{e}")

    # Bottone per inviare la query
    tk.Button(input_window, text="Esegui Query", command=execute_query).grid(row=len(parameter_names) + 2, column=0, columnspan=2, pady=10)

    # Bottone per tornare alla home
    tk.Button(input_window, text="Torna alla Home", command=input_window.destroy).grid(row=len(parameter_names) + 4, column=0, columnspan=2, pady=10)

# Funzione per il secondo tasto
def open_data_screen():
    data_window = tk.Toplevel(root)
    data_window.title("Risultati")

    tk.Label(data_window, text="Risultati Statistici:", font=("Helvetica", 14)).pack(pady=10)

    # Recupera e mostra i dati
    stats_text = tk.Text(data_window, wrap="word", width=80, height=20)
    stats_text.pack(padx=10, pady=10)

    # Visualizza le statistiche
    for stat in matrix:
        stats_text.insert("end", f"Modello: {stat['model']}\n")
        stats_text.insert("end", f"Fase: {stat['phase']}\n")
        stats_text.insert("end", f"Accuratezza: {stat['accuracy']:.4f}\n")
        stats_text.insert("end", "Matrice di Confusione:\n")
        stats_text.insert("end", f"  TN: {stat['confusion_matrix']['TN']}\n")
        stats_text.insert("end", f"  FP: {stat['confusion_matrix']['FP']}\n")
        stats_text.insert("end", f"  FN: {stat['confusion_matrix']['FN']}\n")
        stats_text.insert("end", f"  TP: {stat['confusion_matrix']['TP']}\n")
        stats_text.insert("end", "-" * 40 + "\n")

    stats_text.config(state="disabled")

    # Bottone per chiudere la finestra e tornare alla home
    tk.Button(data_window, text="Torna alla Home", command=data_window.destroy).pack(pady=10)

# Finestra principale
root = tk.Tk()
root.title("Sistema di Classificazione")

tk.Label(root, text="Sistema di Classificazione", font=("Helvetica", 16)).pack(pady=20)

# Tasto 1
tk.Button(root, text="Inserisci Parametri", command=open_input_screen, width=25).pack(pady=10)

# Tasto 2
tk.Button(root, text="Visualizza Risultati", command=open_data_screen, width=25).pack(pady=10)

root.mainloop()
