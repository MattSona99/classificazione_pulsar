import pandas as pd
from pyswip import Prolog
from sklearn import model_selection

from decision_tree import DecisionTree

df = pd.read_csv('pulsar_stars.csv')

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(df.drop('target_class', axis=1), df['target_class'], test_size=0.2)


def train_and_predict_py():

    my_tree_2 = DecisionTree(max_depth=4, min_samples_leaf=1, min_information_gain=0.05)
    my_tree_2.train(X_train, Y_train)

    train_preds = my_tree_2.predict(X_train)
    print("TRAIN PERFORMANCE")
    print("Train size", len(Y_train))
    print("True preds", sum(train_preds == Y_train))
    print("Train accuracy", sum(train_preds == Y_train) / len(Y_train))

    train_preds = my_tree_2.predict(X_test)
    print("TEST PERFORMANCE")
    print("Train size", len(Y_test))
    print("True preds", sum(train_preds == Y_test))
    print("Train accuracy", sum(train_preds == Y_test) / len(Y_test))


def train_and_predict_pl():
    export_to_prolog(X_train, Y_train, 'train_data.pl')
    export_to_prolog(X_test, Y_test, 'test_data.pl')

    prolog = Prolog()
    prolog.consult('decision_tree.pl')
    
    run_tree_query = "run_tree."
    result_run_tree = list(prolog.query(run_tree_query))

    if result_run_tree:
        print("Tree built successfully, now calculating confusion matrix...")

        # Esegui la query che calcola la matrice di confusione
        query = "confusion_matrix(Tree, TestData, TN, FP, FN, TP)."
        result_confusion = list(prolog.query(query))

        # Se la query della matrice di confusione ha restituito dei risultati
        if result_confusion:
            TN = result_confusion[0]['TN']
            FP = result_confusion[0]['FP']
            FN = result_confusion[0]['FN']
            TP = result_confusion[0]['TP']

            # Stampa la matrice di confusione
            print(f"Confusion Matrix:")
            print(f"True Negatives: {TN}")
            print(f"False Positives: {FP}")
            print(f"False Negatives: {FN}")
            print(f"True Positives: {TP}")

            # Calcola e stampa l'accuratezza
            accuracy = (TN + TP) / (TN + FP + FN + TP)
            print(f"Accuracy: {accuracy:.4f}")
        else:
            print("Error: No results found for confusion_matrix.")
    else:
        print("Error: Tree could not be built.")

    

def export_to_prolog(X,Y,filename):
    with open(filename, 'w') as f:
        if(filename=='train_data.pl'):
            for features, label in zip(X.values, Y.values):
                feature_list = ', '.join(map(str, features))
                f.write(f"dato([{feature_list}], {label}).\n")
        else:
            for features, label in zip(X.values, Y.values):
                feature_list = ', '.join(map(str, features))
                f.write(f"dato([{feature_list}], {label}).\n")


train_and_predict_pl()