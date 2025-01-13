:- consult('train_data.pl').
:- consult('test_data.pl').

% Funzione principale
run_tree :-
    % Leggi i dati di training
    findall(dato(Features, Label), dtrain(Features, Label), TrainingData),

    % Costruisci l'albero decisionale
    build_tree(TrainingData, Tree),
   %write('Tree built successfully:'), nl, write(Tree), nl,

   % Potatura dell'albero

    % Leggi i dati di test
    findall(dato(Features, Label), dtest(Features, Label), TestData),

    % Valuta l'albero sui dati di test
    evaluate_tree(Tree, TestData, Accuracy),
    format('Accuracy: ~4f~n', [Accuracy]),

    % Calcola e stampa la matrice di confusione
    confusion_matrix(Tree, TestData),
    halt.

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

% Calcolo dell'entropia
entropy([], 0).
entropy(List, Entropy) :-
    length(List, N),
    findall(P, (member(X, List), count(X, List, Count), P is Count / N), Probabilities),
    maplist(log_entropy, Probabilities, PartialEntropies),
    sumlist(PartialEntropies, Entropy).

log_entropy(P, E) :-
    (P =:= 0 -> E = 0 ; E is -P * log(P) / log(2)).

count(X, List, Count) :-
    include(=(X), List, Matches),
    length(Matches, Count).

% Suddividere i dati in due sottoinsiemi
split([], _, _, [], []).
split([dato(Features, Label) | Rest], AttrIdx, Threshold, Left, Right) :-
    nth1(AttrIdx, Features, AttrValue),
    ( AttrValue < Threshold ->
        Left = [dato(Features, Label) | LeftRest],
        Right = RightRest
    ;
        Left = LeftRest,
        Right = [dato(Features, Label) | RightRest]
    ),
    split(Rest, AttrIdx, Threshold, LeftRest, RightRest).

% Calcolo del guadagno informativo
information_gain(Data, Left, Right, Gain) :-
    findall(Label, member(dato(_, Label), Data), Labels),
    entropy(Labels, EntropyData),
    length(Data, N),
    length(Left, N1),
    length(Right, N2),
    findall(Label, member(dato(_, Label), Left), LeftLabels),
    findall(Label, member(dato(_, Label), Right), RightLabels),
    entropy(LeftLabels, EntropyLeft),
    entropy(RightLabels, EntropyRight),
    WeightedEntropy is (N1 / N) * EntropyLeft + (N2 / N) * EntropyRight,
    Gain is EntropyData - WeightedEntropy.

% Trova la miglior suddivisione (split)
best_split(Data, BestAttrIdx, BestThreshold, BestGain, BestLeft, BestRight) :-
    length(Data, N),
    N > 1,
    Data = [dato(Features, _) | _],
    length(Features, NumFeatures),
    findall((AttrIdx, Threshold, Gain, Left, Right), (
        between(1, NumFeatures, AttrIdx),
        findall(Threshold, (member(dato(Features, _), Data), nth1(AttrIdx, Features, Threshold)), Thresholds),
        list_to_set(Thresholds, UniqueThresholds),
        member(Threshold, UniqueThresholds),
        split(Data, AttrIdx, Threshold, Left, Right),
        length(Left, LLen),
        length(Right, RLen),
        LLen > 0,
        RLen > 0,
        information_gain(Data, Left, Right, Gain)
    ), Splits),
    max_member((BestAttrIdx, BestThreshold, BestGain, BestLeft, BestRight), Splits).

% Costruzione ricorsiva dell'albero decisionale
build_tree([], leaf(none)).
build_tree(Data, leaf(Class)) :-
    findall(Label, member(dato(_, Label), Data), Labels),
    list_to_set(Labels, UniqueLabels),
    length(UniqueLabels, 1),
    UniqueLabels = [Class].

build_tree(Data, node(AttrIdx, Threshold, LeftTree, RightTree)) :-
    best_split(Data, AttrIdx, Threshold, Gain, Left, Right),
    Gain > 0.01,  % Soglia minima per il guadagno informativo
    build_tree(Left, LeftTree),
    build_tree(Right, RightTree).

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
    % Passa i contatori iniziali direttamente come 0
    evaluate_confusion(Tree, TestData, 0, 0, 0, 0, TN, FP, FN, TP),
    format('Confusion Matrix:~n', []),
    format('True Negatives: ~d~n', [TN]),
    format('False Positives: ~d~n', [FP]),
    format('False Negatives: ~d~n', [FN]),
    format('True Positives: ~d~n', [TP]).

% Funzione ricorsiva per confrontare le etichette predette con quelle reali
evaluate_confusion(_, [], TN, FP, FN, TP, TN, FP, FN, TP).  % Caso base: lista vuota
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