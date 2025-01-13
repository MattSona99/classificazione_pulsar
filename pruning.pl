:- consult('train_data.pl').
:- consult('test_data.pl').

% Potatura dell'albero decisionale
prune_tree(leaf(Class), leaf(Class)).  % Caso base: un nodo foglia non viene potato.

prune_tree(node(AttrIdx, Threshold, LeftTree, RightTree), PrunedTree) :-
    % Potatura ricorsiva dei sottoalberi
    prune_tree(LeftTree, PrunedLeft),
    prune_tree(RightTree, PrunedRight),

    % Valutazione del sottoalbero
    ( should_prune(node(AttrIdx, Threshold, PrunedLeft, PrunedRight), PrunedLeft, PrunedRight) ->
        % Se il sottoalbero deve essere potato, convertilo in foglia
        convert_to_leaf(node(AttrIdx, Threshold, PrunedLeft, PrunedRight), PrunedTree)
    ;
        % Altrimenti mantieni il nodo con i sottoalberi potati
        PrunedTree = node(AttrIdx, Threshold, PrunedLeft, PrunedRight)
    ).

% Determina se un sottoalbero deve essere potato
should_prune(Tree, LeftTree, RightTree) :-
    calculate_accuracy(Tree, FullAccuracy),
    calculate_accuracy(leaf_majority(Tree), PrunedAccuracy),
    PrunedAccuracy >= FullAccuracy.  % Potare solo se la precisione non diminuisce.

% Calcola l'accuratezza di un albero
calculate_accuracy(Tree, Accuracy) :-
    findall(dato(Features, Label), dtest(Features, Label), TestData),
    evaluate_samples(Tree, TestData, 0, 0, Total, Correct),
    (Total > 0 -> Accuracy is Correct / Total ; Accuracy is 0).

% Converte un nodo in una foglia basata sulla maggioranza
convert_to_leaf(node(_, _, \, RightTree), leaf(MajorityClass)) :-
    findall(Label, (collect_labels(LeftTree, Label); collect_labels(RightTree, Label)), Labels),
    majority_class(Labels, MajorityClass).

% Raccoglie tutte le etichette da un albero
collect_labels(leaf(Class), Class).
collect_labels(node(_, _, LeftTree, RightTree), Label) :-
    ( collect_labels(LeftTree, Label)
    ; collect_labels(RightTree, Label)
    ).

% Determina la classe di maggioranza
majority_class(Labels, MajorityClass) :-
    msort(Labels, SortedLabels),  % Ordina le etichette
    pack(SortedLabels, Packed),   % Gruppo etichette uguali
    max_member(_-MajorityClass, Packed).

% Raggruppa elementi uguali
pack([], []).
pack([X|Xs], [N-X|Packed]) :-
    take(X, Xs, Ys, Rest),
    length([X|Ys], N),
    pack(Rest, Packed).

% Estrai elementi uguali consecutivi
take(_, [], [], []).
take(X, [Y|Ys], [], [Y|Ys]) :- X \= Y.
take(X, [X|Xs], [X|Ys], Rest) :- take(X, Xs, Ys, Rest).

