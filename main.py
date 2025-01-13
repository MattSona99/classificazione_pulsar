import pandas as pd
from pyswip import Prolog
from sklearn import model_selection
from sklearn.metrics import confusion_matrix

from decision_tree_entropia import DecisionTreeEntropia
from decision_tree_gini import DecisionTreeGini

df = pd.read_csv('pulsar_stars.csv', nrows=1000)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df.drop('target_class', axis=1), 
                                                                    df['target_class'],
                                                                    test_size=0.2,
                                                                    stratify=df['target_class'])
print("Train size", len(Y_train))
print("Test size", len(Y_test))

def train_and_predict_py():

    models = [
        ('Gini', DecisionTreeGini(max_depth=4, min_samples_leaf=1, min_information_gain=0.05)),
        ('Entropia', DecisionTreeEntropia(max_depth=4, min_samples_leaf=1, min_information_gain=0.05))
    ]
    
    for model_name, model in models:
        print(f"==========TRAIN IN PYTHON PER {model_name}===========")
        
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


def train_and_predict_pl():
    export_to_prolog(X_train, Y_train, 'train_data.pl')
    export_to_prolog(X_test, Y_test, 'test_data.pl')
    print("==========TRAIN IN PROLOG===========")

    prolog = Prolog()
    prolog.query("set_prolog_flag(stack_limit, 3*10**9).")

    prolog.consult('decision_tree_entropia.pl')
    
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

def export_tree_to_prolog(tree, filename):
    with open(filename, 'w') as f:
        def write_tree(node):
            if isinstance(node, dict):
                attribute = list(node.keys())[0]
                branches = node[attribute]
                for value, child in branches.items():
                    f.write(f"t({attribute}, [{value}])\n")
                    write_tree(child)
            else:
                f.write(f"l({node})\n")
        write_tree(tree)


train_and_predict_py()
train_and_predict_pl()