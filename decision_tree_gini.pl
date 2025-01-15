:- consult('train_data.pl').
:- consult('test_data.pl').
:- consult('pruning.pl').

% Funzione principale
run_tree(Result) :-
    % Leggi i dati di training
    findall(dato(Features, Label), dtrain(Features, Label), TrainingData),

    % Costruisci l'albero decisionale
    build_tree(TrainingData, Tree),

    % Potatura dell'albero
    prune_tree(Tree, PrunedTree),

    % Leggi i dati di test
    findall(dato(Features, Label), dtest(Features, Label), TestData),

    % Valuta l'albero sui dati di test
    evaluate_tree(PrunedTree, TestData, Accuracy),

    % Calcola la matrice di confusione
    evaluate_confusion(PrunedTree, TestData, 0, 0, 0, 0, TN, FP, FN, TP),

    % Ritorna il risultato come un termine strutturato
    Result = result(accuracy(Accuracy), confusion_matrix(TN, FP, FN, TP)).



% Valutazione dell'albero sui dati di test
evaluate_tree(Tree, TestData, Accuracy) :-
    evaluate_samples(Tree, TestData, 0, 0, Total, Correct),
    Accuracy is Correct / Total.

% Valutazione campione per campione
evaluate_samples(_, [], Total, Correct, Total, Correct).
evaluate_samples(Tree, [dato(Features, TrueLabel) | Rest], TotalAcc, CorrectAcc, Total, Correct) :-
    predict(Tree, dato(Features, TrueLabel), PredictedLabel),
    (PredictedLabel = TrueLabel -> NewCorrect is CorrectAcc + 1 ; NewCorrect is CorrectAcc),
    NewTotal is TotalAcc + 1,
    evaluate_samples(Tree, Rest, NewTotal, NewCorrect, Total, Correct).

% Calcolo del coefficiente di Gini
gini([], 0).
gini(Labels, Gini) :-
    msort(Labels, SortedLabels),
    encode(SortedLabels, Frequencies),
    format('Frequenze: ~w~n', [Frequencies]),
    length(Labels, Total),
    findall(ProbSq, (member(Freq, Frequencies), ProbSq is (Freq / Total) ^ 2), ProbSqs),
    format('Probabilità al quadrato: ~w~n', [ProbSqs]),
    sumlist(ProbSqs, SumProbSqs),
    Gini is 1 - SumProbSqs,
    format('Gini: ~f~n', [Gini]).


square(X, XSquared) :- XSquared is X * X.


count(X, List, Count) :-
    include(=(X), List, Matches),
    length(Matches, Count).

% Suddividere i dati in due sottoinsiemi
split([], _, _, [], []).
split(Examples, Attribute, Threshold, LeftSet, RightSet) :-
    partition(
        [Features, _] >> (nth1(Attribute, Features, Value), Value =< Threshold),
        Examples,
        LeftSet,
        RightSet
    ).


% Calcolo del guadagno di Gini
gini_gain(Set, LeftSet, RightSet, GiniGain) :-
    % Calcola il Gini dell'intero set
    maplist(get_label, Set, Labels),
    gini(Labels, GiniS),
    % Calcola il Gini dei sottoinsiemi
    maplist(get_label, LeftSet, LabelsL),
    gini(LabelsL, GiniSL),
    maplist(get_label, RightSet, LabelsR),
    gini(LabelsR, GiniSR),
    % Calcola le proporzioni dei sottoinsiemi
    length(Set, Total),
    length(LeftSet, TotalL),
    length(RightSet, TotalR),
    PropSL is TotalL / Total,
    PropSR is TotalR / Total,
    % Calcola il Gini Gain
    GiniGain is GiniS - (PropSL * GiniSL + PropSR * GiniSR).


% Trova la miglior suddivisione (split)
% Funzione per trovare lo split migliore
best_split(Examples, Attribute, BestThreshold, BestGain, LeftSet, RightSet) :-
    findall(Value, (member([Features, _], Examples), nth1(Attribute, Features, Value)), Values),
    sort(Values, SortedValues),
    findall(
        Gain-Threshold-Left-Right,
        (member(Threshold, SortedValues),
         split(Examples, Attribute, Threshold, Left, Right),
         gini_gain(Examples, Left, Right, Gain)),
        GainsThresholdsSubsets
    ),
    max_member(BestGain-BestThreshold-LeftSet-RightSet, GainsThresholdsSubsets).


% Costruzione ricorsiva dell'albero decisionale
build_tree([], leaf(none)).
build_tree(Data, leaf(Class)) :-
    % Trova tutte le etichette dei dati
    findall(Label, member(dato(_, Label), Data), Labels),
    % Conta le etichette uniche
    list_to_set(Labels, UniqueLabels),
    length(UniqueLabels, NumClasses),
    % Se c'è solo una classe, crea una foglia
    ( NumClasses = 1 ->
        UniqueLabels = [Class]
    ;
        % Altrimenti, crea una foglia con la classe più comune
        aggregate(max(Count, Label), count(Label, Labels, Count), max(_, Class))
    ).


build_tree(Data, node(AttrIdx, Threshold, LeftTree, RightTree)) :-
    best_split(Data, AttrIdx, Threshold, Gain, LeftSet, RightSet),
    Gain > 0,  % Controlla che il guadagno sia positivo
    build_tree(LeftSet, LeftTree),
    build_tree(RightSet, RightTree).


% Predizione
predict(leaf(Class), _, Class).
predict(node(AttrIdx, Threshold, LeftTree, RightTree), dato(Features, _), Class) :-
    nth1(AttrIdx, Features, Value),
    ( Value < Threshold ->
        predict(LeftTree, dato(Features, _), Class)
    ;
        predict(RightTree, dato(Features, _), Class)
    ).

% Funzione per calcolare la matrice di confusione
confusion_matrix(Tree, TestData) :-
    evaluate_confusion(Tree, TestData, 0, 0, 0, 0, TN, FP, FN, TP),
    format('Confusion Matrix:~n', []),
    format('True Negatives: ~d~n', [TN]),
    format('False Positives: ~d~n', [FP]),
    format('False Negatives: ~d~n', [FN]),
    format('True Positives: ~d~n', [TP]).

% Funzione ricorsiva per confrontare le etichette predette con quelle reali
evaluate_confusion(_, [], TN, FP, FN, TP, TN, FP, FN, TP).
evaluate_confusion(Tree, [dato(Features, TrueLabel) | Rest], TN, FP, FN, TP, NewTN, NewFP, NewFN, NewTP) :-
    predict(Tree, dato(Features, _), PredictedLabel),
    update_counts(PredictedLabel, TrueLabel, TN, FP, FN, TP, TempTN, TempFP, TempFN, TempTP),
    evaluate_confusion(Tree, Rest, TempTN, TempFP, TempFN, TempTP, NewTN, NewFP, NewFN, NewTP).

% Funzione per aggiornare i contatori della matrice di confusione
update_counts(PredictedLabel, TrueLabel, TN, FP, FN, TP, NewTN, NewFP, NewFN, NewTP) :-
    (PredictedLabel = 0, TrueLabel = 0 -> NewTN is TN + 1; NewTN is TN),
    (PredictedLabel = 1, TrueLabel = 0 -> NewFP is FP + 1; NewFP is FP),
    (PredictedLabel = 0, TrueLabel = 1 -> NewFN is FN + 1; NewFN is FN),
    (PredictedLabel = 1, TrueLabel = 1 -> NewTP is TP + 1; NewTP is TP).

% Classificazione di un singolo oggetto
classify_object(Tree, Features, PredictedClass) :-
    % Crea un dato fittizio senza etichetta per effettuare la predizione
    dato(Features, _) = Object,
    predict(Tree, Object, PredictedClass).

% Predizione dinamica basata su input da Python
classify_example(InputFeatures, PredictedClass) :-
    % Leggi i dati di training e costruisci l'albero
    findall(dato(Features, Label), dtrain(Features, Label), TrainingData),
    build_tree(TrainingData, Tree),
    prune_tree(Tree, PrunedTree),
    % Predizione usando le caratteristiche passate da Python
    classify_object(PrunedTree, InputFeatures, PredictedClass).
