/**
 * author: Samuel Kuchta
 * login: xkucht11
 * date: 24.4.2025
 * project: flp24-log, rubik's cube
*/


% Entry point of the program. Reads cube state from input,
% parses it, prints it, checks if it is already solved, and solves if needed.
% includes validity testing (optional)
start :-
    prompt(_, ''),
    read_lines(Lines),
    split_lines(Lines, SplitLines),
    parse_cube(SplitLines, Cube),
    print_cube(Cube),
    (cube_is_solved(Cube) -> halt ; 
        %test_moves, 
        solve_with_ids(Cube, Solution), 
        print_solution(Cube, Solution)),
    halt.


% Reads all lines from input until EOF
read_lines(Ls) :-
    read_line(L, C),
    (C == end_of_file, Ls = [] ;
      read_lines(LLs), Ls = [L|LLs]
    ).

% Reads a single line of characters until EOL or EOF
read_line(L, C) :-
    get_char(C),
    (isEOFEOL(C), L = [], !;
        read_line(LL, _),
        [C|LL] = L).

% Helper to check if character is EOF or newline
isEOFEOL(C) :-
    C == end_of_file;
    (char_code(C, Code), Code==10).

% Splits all lines into groups of characters
split_lines([],[]).
split_lines([L|Ls],[H|T]) :- split_lines(Ls, T), split_line(L, H).

% Splits a single line into groups separated by spaces
split_line([],[[]]) :- !.
split_line([' '|T], [[]|S1]) :- !, split_line(T, S1).
split_line([32|T], [[]|S1]) :- !, split_line(T, S1).
split_line([H|T], [[H|G]|S1]) :- split_line(T,[G|S1]).

% Gets the Nth element from a list
nth_elem([X|_], 1, X).
nth_elem([_|Tail], N, Result) :- 
    N > 1,
    N1 is N - 1,
    nth_elem(Tail, N1, Result).
    
% Flattens a list of lists into a single list
flatten_list([], []).
flatten_list([X|XS], Result) :-
    flatten_list(XS, Rest),
    append(X, Rest, Result).


% Converts split line groups into cube representation
parse_cube([
        [Up1], [Up2], [Up3], 
        Mid_High, Mid_Center, Mid_Low, 
        [Down1], [Down2], [Down3]
    ], Cube) :-

    nth_elem(Mid_High, 1, Front1), 
    nth_elem(Mid_Center, 1, Front2), 
    nth_elem(Mid_Low, 1, Front3),

    nth_elem(Mid_High, 2, Right1), 
    nth_elem(Mid_Center, 2, Right2), 
    nth_elem(Mid_Low, 2, Right3),

    nth_elem(Mid_High, 3, Back1), 
    nth_elem(Mid_Center, 3, Back2), 
    nth_elem(Mid_Low, 3, Back3),

    nth_elem(Mid_High, 4, Left1), 
    nth_elem(Mid_Center, 4, Left2), 
    nth_elem(Mid_Low, 4, Left3),

    flatten_list([Front1, Front2, Front3], Front_Side), 
    flatten_list([Right1, Right2, Right3], Right_Side),
    flatten_list([Back1, Back2, Back3], Back_Side), 
    flatten_list([Left1, Left2, Left3], Left_Side), 
    flatten_list([Up1, Up2, Up3], Up_Side),
    flatten_list([Down1, Down2, Down3], Down_Side),

    Cube = [
            Front_Side, Right_Side, Back_Side, Left_Side, 
            Up_Side, Down_Side
        ].



% Prints the cube state in human-readable format
print_cube([
            [Front11, Front12, Front13, Front21, Front22, Front23, Front31, Front32, Front33],
            [Right11, Right12, Right13, Right21, Right22, Right23, Right31, Right32, Right33],
            [Back11, Back12, Back13, Back21, Back22, Back23, Back31, Back32, Back33],
            [Left11, Left12, Left13, Left21, Left22, Left23, Left31, Left32, Left33],
            Up,
            Down
        ]) :-

    format('~w~w~w\n~w~w~w\n~w~w~w\n', Up),

    format('~w~w~w ~w~w~w ~w~w~w ~w~w~w\n', [
        Front11, Front12, Front13, 
        Right11, Right12, Right13, 
        Back11, Back12, Back13,
        Left11, Left12, Left13
    ]),
    format('~w~w~w ~w~w~w ~w~w~w ~w~w~w\n', [
        Front21, Front22, Front23, 
        Right21, Right22, Right23, 
        Back21, Back22, Back23,
        Left21, Left22, Left23
    ]),
    format('~w~w~w ~w~w~w ~w~w~w ~w~w~w\n', [
        Front31, Front32, Front33, 
        Right31, Right32, Right33, 
        Back31, Back32, Back33,
        Left31, Left32, Left33 
    ]),

    format('~w~w~w\n~w~w~w\n~w~w~w\n', Down).




% Verifies if all faces are completely solved (single color per face)
cube_is_solved(Cube) :-
    forall(member(Face, Cube), 
    sort(Face, [_])). % All elements in face are identical


% Solves cube using Iterative Deepening Search
solve_with_ids(Cube, Solution) :-
    iterative_deepening_search(Cube, 0, 20, Solution).

iterative_deepening_search(Cube, CurrentDepth, MaxDepth, Solution) :-
    CurrentDepth =< MaxDepth,
    depth_limited_search(Cube, CurrentDepth, [], Solution),
    !.

iterative_deepening_search(Cube, CurrentDepth, MaxDepth, Solution) :-
    CurrentDepth < MaxDepth,
    NextDepth is CurrentDepth + 1,
    iterative_deepening_search(Cube, NextDepth, MaxDepth, Solution).

depth_limited_search(Cube, 0, _, []) :-
    cube_is_solved(Cube).

depth_limited_search(Cube, Depth, Path, [Move|Solution]) :-
    Depth > 0,
    NewDepth is Depth - 1,
    valid_moves(Moves),
    filter_inverse_moves(Moves, Path, ValidMoves),
    member(Move, ValidMoves),
    move(Move, Cube, NewCube),
    depth_limited_search(NewCube, NewDepth, [Move|Path], Solution).

% Removes moves that would undo previous step
filter_inverse_moves([], _, []).
filter_inverse_moves(Moves, [Last|_], ValidMoves) :-
    opposite_move(Last, Opposite),
    delete(Moves, Opposite, ValidMoves),
    !.
filter_inverse_moves(Moves, [], Moves).



% All possible cube moves including slice turns
valid_moves([u, uPrime, d, dPrime, r, rPrime, l, lPrime, f, fPrime, b, bPrime, m, mPrime, e, ePrime, s, sPrime]).

% Define inverse relationships between moves
opposite_move(u, uPrime).
opposite_move(uPrime, u).
opposite_move(d, dPrime).
opposite_move(dPrime, d).
opposite_move(r, rPrime).
opposite_move(rPrime, r).
opposite_move(l, lPrime).
opposite_move(lPrime, l).
opposite_move(f, fPrime).
opposite_move(fPrime, f).
opposite_move(b, bPrime).
opposite_move(bPrime, b).
opposite_move(m, mPrime).
opposite_move(mPrime, m).
opposite_move(e, ePrime).
opposite_move(ePrime, e).
opposite_move(s, sPrime).
opposite_move(sPrime, s).


move(Move, [
        [F11, F12, F13, F21, F22, F23, F31, F32, F33],
        [R11, R12, R13, R21, R22, R23, R31, R32, R33],
        [B11, B12, B13, B21, B22, B23, B31, B32, B33],
        [L11, L12, L13, L21, L22, L23, L31, L32, L33],
        [U11, U12, U13, U21, U22, U23, U31, U32, U33],
        [D11, D12, D13, D21, D22, D23, D31, D32, D33]
        ], NewCube) :-
    (
    Move == u, 
    NewU = [U31, U21, U11, U32, U22, U12, U33, U23, U13], % move U face clockwise
    NewCube = [
        [R11, R12, R13, F21, F22, F23, F31, F32, F33], % Front top from R's top
        [B11, B12, B13, R21, R22, R23, R31, R32, R33], % R top from B's top
        [L11, L12, L13, B21, B22, B23, B31, B32, B33], % B top from L's top
        [F11, F12, F13, L21, L22, L23, L31, L32, L33], % L top from F's top
        NewU,
        [D11, D12, D13, D21, D22, D23, D31, D32, D33]  % D remains unchanged
    ]);
    (
    Move == uPrime, 
    NewU = [U13, U23, U33, U12, U22, U32, U11, U21, U31], % move U face counter-clockwise
    NewCube = [
        [L11, L12, L13, F21, F22, F23, F31, F32, F33], % Front top from L's top
        [F11, F12, F13, R21, R22, R23, R31, R32, R33], % R top from F's top
        [R11, R12, R13, B21, B22, B23, B31, B32, B33], % B top from R's top
        [B11, B12, B13, L21, L22, L23, L31, L32, L33], % L top from B's top
        NewU,
        [D11, D12, D13, D21, D22, D23, D31, D32, D33]  % D remains unchanged
    ]);
    (
    Move == d, 
    NewD = [D31, D21, D11, D32, D22, D12, D33, D23, D13], % move D face clockwise
    NewCube = [
        [F11, F12, F13, F21, F22, F23, L31, L32, L33], % Front bottom from L's bottom
        [R11, R12, R13, R21, R22, R23, F31, F32, F33], % R bottom from F's bottom
        [B11, B12, B13, B21, B22, B23, R31, R32, R33], % B bottom from R's bottom
        [L11, L12, L13, L21, L22, L23, B31, B32, B33], % L bottom from B's bottom
        [U11, U12, U13, U21, U22, U23, U31, U32, U33], % U remains unchanged
        NewD
    ]);
    (
    Move == dPrime, 
    NewD = [D13, D23, D33, D12, D22, D32, D11, D21, D31], % move D face counter-clockwise
    NewCube = [
        [F11, F12, F13, F21, F22, F23, R31, R32, R33], % Front bottom from R's bottom
        [R11, R12, R13, R21, R22, R23, B31, B32, B33], % R bottom from B's bottom
        [B11, B12, B13, B21, B22, B23, L31, L32, L33], % B bottom from L's bottom
        [L11, L12, L13, L21, L22, L23, F31, F32, F33], % L bottom from F's bottom
        [U11, U12, U13, U21, U22, U23, U31, U32, U33], % U remains unchanged
        NewD
    ]);
    (
    Move == r, 
    NewR = [R31, R21, R11, R32, R22, R12, R33, R23, R13], % move R face clockwise
    NewCube = [
        [F11, F12, D13, F21, F22, D23, F31, F32, D33],  % Front right from Down's right column
        NewR,
        [U33, B12, B13, U23, B22, B23, U13, B32, B33],  % Back left from Up's right reversed
        [L11, L12, L13, L21, L22, L23, L31, L32, L33],  % L remains unchanged
        [U11, U12, F13, U21, U22, F23, U31, U32, F33],  % Up right from Front's right column
        [D11, D12, B31, D21, D22, B21, D31, D32, B11]   % Down right from Back's left reversed
    ]);
    (
    Move == rPrime, 
    NewR = [R13, R23, R33, R12, R22, R32, R11, R21, R31], % move R face counter-clockwise
    NewCube = [
        [F11, F12, U13, F21, F22, U23, F31, F32, U33],  % Front right from Up's right column
        NewR,
        [D33, B12, B13, D23, B22, B23, D13, B32, B33],  % Back left from Down's right reversed
        [L11, L12, L13, L21, L22, L23, L31, L32, L33],  % L remains unchanged
        [U11, U12, B31, U21, U22, B21, U31, U32, B11],  % Up right from Back's left reversed
        [D11, D12, F13, D21, D22, F23, D31, D32, F33]   % Down right from Front's right column
    ]);
    (
    Move == l, 
    NewL = [L31, L21, L11, L32, L22, L12, L33, L23, L13], % move L face clockwise
    NewCube = [
        [U11, F12, F13, U21, F22, F23, U31, F32, F33],  % Front left from Up's left column
        [R11, R12, R13, R21, R22, R23, R31, R32, R33],  % R remains unchanged
        [B11, B12, D31, B21, B22, D21, B31, B32, D11],  % Back right from Down's left reversed
        NewL,
        [B33, U12, U13, B23, U22, U23, B13, U32, U33],  % Up left from Back's right reversed
        [F11, D12, D13, F21, D22, D23, F31, D32, D33]   % Down left from Front's left column
    ]);
    (
    Move == lPrime,
    NewL = [L13, L23, L33, L12, L22, L32, L11, L21, L31], % move L face counter-clockwise
    NewCube = [
        [D11, F12, F13, D21, F22, F23, D31, F32, F33],   % Front left <- Down left column
        [R11, R12, R13, R21, R22, R23, R31, R32, R33],   % R remains unchanged
        [B11, B12, U11, B21, B22, U21, B31, B32, U31],   % Back right <- Up left (reversed)
        NewL,                                             % Updated L face
        [F11, U12, U13, F21, U22, U23, F31, U32, U33],   % Up left <- Front left (original)
        [B13, D12, D13, B23, D22, D23, B33, D32, D33]    % Down left <- Back right (reversed)
    ]);
    (
    Move == f, 
    NewF = [F31, F21, F11, F32, F22, F12, F33, F23, F13], % move F face clockwise
    NewCube = [
        NewF,
        [U31, R12, R13, U32, R22, R23, U33, R32, R33],  % R left from Up's bottom row
        [B11, B12, B13, B21, B22, B23, B31, B32, B33],  % B remains unchanged
        [L11, L12, D11, L21, L22, D12, L31, L32, D13],  % L right from Down's top reversed
        [U11, U12, U13, U21, U22, U23, L33, L23, L13],  % Up bottom from L's right reversed
        [R31, R21, R11, D21, D22, D23, D31, D32, D33]   % Down top from R's left column
    ]);
    (
    Move == fPrime, 
    NewF = [F13, F23, F33, F12, F22, F32, F11, F21, F31], % move F face counter-clockwise
    NewCube = [
        NewF,
        [D13, R12, R13, D12, R22, R23, D11, R32, R33],  % R left from Down's top reversed
        [B11, B12, B13, B21, B22, B23, B31, B32, B33],  % B remains unchanged
        [L11, L12, U33, L21, L22, U32, L31, L32, U31],  % L right from Up's bottom row
        [U11, U12, U13, U21, U22, U23, R11, R21, R31],  % Up bottom from R's left column
        [L13, L23, L33, D21, D22, D23, D31, D32, D33]   % Down top from L's right reversed
    ]);
    (
    Move == b, 
    NewB = [B31, B21, B11, B32, B22, B12, B33, B23, B13], % move B face clockwise
    NewCube = [
        [F11, F12, F13, F21, F22, F23, F31, F32, F33],  % F remains unchanged
        [R11, R12, D33, R21, R22, D32, R31, R32, D31],  % R right from Down's bottom reversed
        NewB,
        [U13, L12, L13, U12, L22, L23, U11, L32, L33],  % L left from Up's top reversed
        [R13, R23, R33, U21, U22, U23, U31, U32, U33],  % Up top from R's right column
        [D11, D12, D13, D21, D22, D23, L11, L21, L31]   % Down bottom from L's left column
    ]);
    (
    Move == bPrime, 
    NewB = [B13, B23, B33, B12, B22, B32, B11, B21, B31], % move B face counter-clockwise
    NewCube = [
        [F11, F12, F13, F21, F22, F23, F31, F32, F33],  % F remains unchanged
        [R11, R12, U11, R21, R22, U12, R31, R32, U13],  % R right from Up's top reversed
        NewB,
        [D31, L12, L13, D32, L22, L23, D33, L32, L33],  % L left from Down's bottom reversed
        [L31, L21, L11, U21, U22, U23, U31, U32, U33],  % Up top from L's left column
        [D11, D12, D13, D21, D22, D23, R33, R23, R13]   % Down bottom from R's right reversed
    ]);
    (
    Move == m, 
    NewCube = [
        [F11, U12, F13, F21, U22, F23, F31, U32, F33], % Front middle from Up's middle
        [R11, R12, R13, R21, R22, R23, R31, R32, R33], % R remains unchanged
        [B11, D32, B13, B21, D22, B23, B31, D12, B33], % Back middle from Down's middle
        [L11, L12, L13, L21, L22, L23, L31, L32, L33], % L remains unchanged
        [U11, B32, U13, U21, B22, U23, U31, B12, U33], % Up middle from Back's middle
        [D11, F12, D13, D21, F22, D23, D31, F32, D33]  % Down middle from Front's middle
    ]);
    (
    Move == mPrime, 
    NewCube = [
        [F11, D12, F13, F21, D22, F23, F31, D32, F33], % Front middle from Down's middle
        [R11, R12, R13, R21, R22, R23, R31, R32, R33], % R remains unchanged
        [B11, U12, B13, B21, U22, B23, B31, U32, B33], % Back middle from Up's middle
        [L11, L12, L13, L21, L22, L23, L31, L32, L33], % L remains unchanged
        [U11, F12, U13, U21, F22, U23, U31, F32, U33], % Up middle from Front's middle
        [D11, B12, D13, D21, B22, D23, D31, B32, D33]  % Down middle from Back's middle
    ]);
    (
    Move == e, 
    NewCube = [
        [F11, F12, F13, L21, L22, L23, F31, F32, F33], % Front middle from L's middle
        [R11, R12, R13, F21, F22, F23, R31, R32, R33], % R middle from F's middle
        [B11, B12, B13, R21, R22, R23, B31, B32, B33], % B middle from R's middle
        [L11, L12, L13, B21, B22, B23, L31, L32, L33], % L middle from B's middle
        [U11, U12, U13, U21, U22, U23, U31, U32, U33], % U remains unchanged
        [D11, D12, D13, D21, D22, D23, D31, D32, D33] % D remains unchanged
    ]);
    (
    Move == ePrime, 
    NewCube = [
        [F11, F12, F13, R21, R22, R23, F31, F32, F33], % Front middle from R's middle
        [R11, R12, R13, B21, B22, B23, R31, R32, R33], % R middle from B's middle
        [B11, B12, B13, L21, L22, L23, B31, B32, B33], % B middle from L's middle
        [L11, L12, L13, F21, F22, F23, L31, L32, L33], % L middle from F's middle
        [U11, U12, U13, U21, U22, U23, U31, U32, U33], % U remains unchanged
        [D11, D12, D13, D21, D22, D23, D31, D32, D33] % D remains unchanged
    ]);
    (
    Move == s, 
    NewCube = [
        [F11, F12, F13, F21, F22, F23, F31, F32, F33], % F remains unchanged
        [R11, U21, R13, R21, U22, R23, R31, U23, R33], % R middle from Up's middle
        [B11, B12, B13, B21, B22, B23, B31, B32, B33], % B remains unchanged
        [L11, D21, L13, L21, D22, L23, L31, D23, L33], % L middle from Down's middle
        [U11, U12, U13, L32, L22, L12, U31, U32, U33], % Up middle from L's right and left
        [D11, D12, D13, R32, R22, R12, D31, D32, D33] % Down middle from R's right and left
    ]);
    (
    Move == sPrime, 
    NewCube = [
        [F11, F12, F13, F21, F22, F23, F31, F32, F33], % F remains unchanged
        [R11, D23, R13, R21, D22, R23, R31, D21, R33], % R middle from Down's middle
        [B11, B12, B13, B21, B22, B23, B31, B32, B33], % B remains unchanged
        [L11, U23, L13, L21, U22, L23, L31, U21, L33], % L middle from Up's middle
        [U11, U12, U13, R12, R22, R32, U31, U32, U33], % Up middle from R's right and left
        [D11, D12, D13, L12, L22, L32, D31, D32, D33] % Down middle from L's right and left
    ]).


% Output
print_solution(_, []).
print_solution(Cube, [Move|Moves]) :-
    move(Move, Cube, NewCube),
    nl,
    print_cube(NewCube),
    print_solution(NewCube, Moves).











/*
% Testing predicate to validate all moves
test_moves :-
    initial_solved_cube(InitialCube),
    valid_moves(Moves),
    forall(member(Move, Moves), (
        format('TEST Move ~w\n', [Move]),
        test_four_moves(Move, InitialCube),
        test_move_inverse(Move, InitialCube)
    )),
    test_specific_sequence,
    writeln('All move tests passed.').


% Tests if applying the specific sequence and its inverse returns to solved
test_specific_sequence :-
    initial_solved_cube(InitialCube),
    %Sequence = [r, u, rPrime, fPrime, u, f, uPrime, rPrime, f, r, fPrime, uPrime],
    %Sequence = [e, f, fPrime, ePrime],
    Sequence = [m, e, s, l, e, 
                r, u, rPrime, fPrime, u, f, uPrime, rPrime, f, r, fPrime, uPrime,
                f, r, u, b,
                r, u, rPrime, fPrime, u, f, uPrime, rPrime, f, r, fPrime, uPrime, 
                bPrime, uPrime, rPrime, fPrime, 
                r, u, rPrime, fPrime, u, f, uPrime, rPrime, f, r, fPrime, uPrime,
                ePrime, lPrime, sPrime, ePrime, mPrime],


    apply_moves(Sequence, InitialCube, AfterSeq),
    ( cube_is_solved(AfterSeq) ->
        writeln('Specific sequence test passed.')
    ;
        writeln('Specific sequence test FAILED.')
    ),
    inverse_sequence(Sequence, InverseSeq),
    apply_moves(InverseSeq, AfterSeq, FinalCube),
    ( cube_is_solved(FinalCube) ->
        writeln('Specific sequence inverse test passed.')
    ;
        writeln('Specific sequence inverse test FAILED.')
    ).

inverse_sequence(Seq, InvSeq) :-
    reverse(Seq, RevSeq),
    maplist(opposite_move, RevSeq, InvSeq).


% Creates an initial solved cube state for testing
initial_solved_cube([
    ['F','F','F','F','F','F','F','F','F'],
    ['R','R','R','R','R','R','R','R','R'],
    ['B','B','B','B','B','B','B','B','B'],
    ['L','L','L','L','L','L','L','L','L'],
    ['U','U','U','U','U','U','U','U','U'],
    ['D','D','D','D','D','D','D','D','D']
]).

% Tests if applying a move four times returns to original state
test_four_moves(Move, InitialCube) :-
    apply_moves([Move, Move, Move, Move], InitialCube, FinalCube),
    ( cube_is_solved(FinalCube)
    -> true
    ;  format('FAIL: Four ~w moves do not return to solved state.~n', [Move]),
       true
    ).

% Tests if a move followed by its inverse returns to original state
test_move_inverse(Move, InitialCube) :-
    opposite_move(Move, Inverse),
    apply_moves([Move, Inverse], InitialCube, FinalCube),
    ( cube_is_solved(FinalCube)
    -> true
    ;  format('FAIL: Move ~w followed by ~w does not return to solved.~n', [Move, Inverse]),
       true
    ).

% Applies a list of moves sequentially to a cube
apply_moves([], Cube, Cube).
apply_moves([Move|Moves], Cube, Result) :-
    move(Move, Cube, NewCube),
    apply_moves(Moves, NewCube, Result).
*/