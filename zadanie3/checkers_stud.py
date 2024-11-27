#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji. 
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.
Mam świadomość wielu jego braków ale nie mam czasu na jego poprawianie.

Zasady gry: https://en.wikipedia.org/wiki/English_draughts (w skrócie: wszyscy ruszają się po 1 polu. Pionki tylko w kierunku wroga, damki w dowolnym)
  z następującymi modyfikacjami: a) bicie nie jest wymagane,  b) dozwolone jest tylko pojedyncze bicie (bez serii).

Należy napisać funkcje "minimax_a_b_recurr", "minimax_a_b" (która woła funkcję rekurencyjną) i funkcje "*ev_func", która oceniają stan gry

Chętni mogą ulepszać mój kod (trzeba oznaczyć komentarzem co zostało zmienione), mogą również dodać obsługę bicia wielokrotnego i wymagania bicia. Mogą również wdrożyć reguły: https://en.wikipedia.org/wiki/Russian_draughts
"""

import numpy as np
import pygame
from copy import deepcopy
import pandas as pd
import multiprocessing as mp

FPS = 20

MINIMAX_DEPTH = 5

WIN_WIDTH = 800
WIN_HEIGHT = 800

WON_PRIZE = 10000

MOVES_HIST_LEN = 6

BOARD_WIDTH = BOARD_HEIGHT = 8

FIELD_SIZE = WIN_WIDTH / BOARD_WIDTH
PIECE_SIZE = FIELD_SIZE / 2 - 8
MARK_THICK = 2
POS_MOVE_MARK_SIZE = PIECE_SIZE / 2


BLACK_PIECES_COL = (0, 0, 0)
WHITE_PIECES_COL = (255, 255, 255)
POSS_MOVE_MARK_COL = (255, 0, 0)
DARK_BOARD_COL = (196, 164, 132)
BRIGHT_BOARD_COL = (250, 250, 250)
KING_MARK_COL = (255, 215, 0)


# count difference between the number of pieces, king+10
def basic_ev_func(board: "Board", is_black_turn: bool) -> int:
    h = 0
    pawn = 1
    king = 10
    # ToDo funkcja liczy i zwraca ocene aktualnego stanu planszy
    if board.white_fig_left == 0:
        return -WON_PRIZE
    if board.black_fig_left == 0:
        return WON_PRIZE
    for row in range(BOARD_WIDTH):
        for col in range((row + 1) % 2, BOARD_WIDTH, 2):
            if board.board[row][col].is_white():
                if board.board[row][col].is_king():
                    h += king
                else:
                    h += pawn
            elif board.board[row][col].is_black():
                if board.board[row][col].is_king():
                    h -= king
                else:
                    h -= pawn
    # board.board[row][col].is_black() - sprawdza czy to czarny kolor figury
    # board.board[row][col].is_white() - sprawdza czy to biały kolor figury
    # board.board[row][col].is_king() - sprawdza czy to damka
    # współrzędne zaczynają (0,0) się od lewej od góry
    return h


# nagrody jak w wersji podstawowej + nagroda za stopień zwartości grupy
def group_prize_ev_func(board: "Board", is_black_turn: bool) -> int:
    h = 0
    pawn = 1
    king = 10
    neighbour = 1
    edge = 1
    if board.white_fig_left == 0:
        return -WON_PRIZE
    if board.black_fig_left == 0:
        return WON_PRIZE
    for row in range(BOARD_WIDTH):
        for col in range((row + 1) % 2, BOARD_WIDTH, 2):
            piece = board.board[row][col]
            if piece.is_white():
                if piece.is_king():
                    h += king
                else:
                    h += pawn
                neighbors = sum(
                    neighbour
                    for i in range(-1, 2, 2)
                    for j in range(-1, 2, 2)
                    if 0 <= row + i < len(board.board)
                    and 0 <= col + j < len(board.board[row])
                    and board.board[row + i][col + j].is_white()
                )
                h += neighbors
                if (
                    col == 0
                    or col == len(board.board[row]) - 1
                    or row == 0
                    or row == len(board.board) - 1
                ):
                    h += edge

            elif piece.is_black():
                if piece.is_king():
                    h -= king
                else:
                    h -= pawn
                neighbors = sum(
                    neighbour
                    for i in range(-1, 2, 2)
                    for j in range(-1, 2, 2)
                    if 0 <= row + i < len(board.board)
                    and 0 <= col + j < len(board.board[row])
                    and board.board[row + i][col + j].is_black()
                )
                h -= neighbors
                if (
                    col == 0
                    or col == len(board.board[row]) - 1
                    or row == 0
                    or row == len(board.board) - 1
                ):
                    h -= edge
    return h


# za każdy pion na własnej połowie planszy otrzymuje się 5 nagrody, na połowie przeciwnika 7, a za każdą damkę 10.
def push_to_opp_half_ev_func(board: "Board", is_black_turn: bool) -> int:
    h = 0
    king = 10
    own_half = 5
    opp_half = 7
    # ToDo
    if board.white_fig_left == 0:
        return -WON_PRIZE
    if board.black_fig_left == 0:
        return WON_PRIZE
    for row in range(BOARD_WIDTH):
        for col in range((row + 1) % 2, BOARD_WIDTH, 2):
            if board.board[row][col].is_white():
                if board.board[row][col].is_king():
                    h += king
                else:
                    if row >= len(board.board) // 2:
                        h += own_half
                    else:
                        h += opp_half
            elif board.board[row][col].is_black():
                if board.board[row][col].is_king():
                    h -= king
                else:
                    if row < len(board.board) // 2:
                        h -= own_half
                    else:
                        h -= opp_half
    return h


# za każdy nasz pion otrzymuje się nagrodę w wysokości: (5 + numer wiersza, na którym stoi pion) (im jest bliżej wroga tym lepiej), a za każdą damkę dodtakowe: 10.
def push_forward_ev_func(board: "Board", is_black_turn: bool) -> int:
    h = 0
    king = 10
    pawn = 5
    # ToDo
    if board.white_fig_left == 0:
        return -WON_PRIZE
    if board.black_fig_left == 0:
        return WON_PRIZE
    for row in range(BOARD_WIDTH):
        for col in range((row + 1) % 2, BOARD_WIDTH, 2):
            if board.board[row][col].is_white():
                if board.board[row][col].is_king():
                    h += king
                else:
                    h += pawn + (BOARD_WIDTH - row - 1)
            elif board.board[row][col].is_black():
                if board.board[row][col].is_king():
                    h -= king
                else:
                    h -= pawn + row
    return h


# f. called from main
def minimax_a_b(board: "Board", depth: int, plays_as_black: bool, ev_func: callable):
    possible_moves = board.get_possible_moves(plays_as_black)
    if len(possible_moves) == 0:
        board.white_won = plays_as_black
        board.is_running = False
        return None
    a = -np.inf
    b = np.inf
    moves_marks = []
    for possible_move in possible_moves:
        # ToDo
        board_copy = deepcopy(board)
        ev = minimax_a_b_recurr(
            board_copy, depth, not plays_as_black, a, b, ev_func, [possible_move]
        )
        moves_marks.append(ev)
    if not plays_as_black:
        best_indices = np.where(moves_marks == np.max(moves_marks))[0]
    else:
        best_indices = np.where(moves_marks == np.min(moves_marks))[0]
    return possible_moves[np.random.choice(best_indices)]


# recursive function, called from minimax_a_b
def minimax_a_b_recurr(
    board: "Board",
    depth: int,
    move_max: bool,
    a: float,
    b: float,
    ev_func: callable,
    possible_moves: list["Move"] = None,
) -> float:
    # ToDo
    if possible_moves is None:
        possible_moves = board.get_possible_moves(not move_max)

    if len(possible_moves) == 0 or depth == 0:
        return ev_func(board, not move_max)

    if move_max:
        for move in possible_moves:
            board_copy = deepcopy(board)
            board_copy.make_move(move)
            a = max(
                a,
                minimax_a_b_recurr(board_copy, depth - 1, not move_max, a, b, ev_func),
            )
            if a >= b:
                break
        return a
    else:
        for move in possible_moves:
            board_copy = deepcopy(board)
            board_copy.make_move(move)
            b = min(
                b,
                minimax_a_b_recurr(board_copy, depth - 1, not move_max, a, b, ev_func),
            )
            if a >= b:
                break
        # return value
        return b


class Move:
    def __init__(self, piece, dest_row, dest_col, captures=None):
        self.piece = piece
        self.dest_row = dest_row
        self.dest_col = dest_col
        self.captures = captures

    def __eq__(self, other):
        if other is None:
            return False
        return (
            self.piece == other.piece
            and self.dest_row == other.dest_row
            and self.dest_col == other.dest_col
            and self.captures == other.captures
        )

    def __str__(self):
        return (
            "Move from r, c:"
            + str(self.piece.row)
            + ", "
            + str(self.piece.col)
            + ", to:"
            + str(self.dest_row)
            + ", "
            + str(self.dest_col)
            + ", "
            + str(id(self.piece))
        )


class Field:
    def is_empty(self):
        return True

    def is_white(self):
        return False

    def is_black(self):
        return False

    def __str__(self):
        return "."


class Pawn(Field):
    def __init__(self, is_white, row, col):
        self.__is_white = is_white
        self.row = row
        self.col = col

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def __str__(self):
        if self.is_white():
            return "w"
        return "b"

    def is_king(self):
        return False

    def is_empty(self):
        return False

    def is_white(self):
        return self.__is_white

    def is_black(self):
        return not self.__is_white


class King(Pawn):
    def __init__(self, pawn):
        super().__init__(pawn.is_white(), pawn.row, pawn.col)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        return result

    def is_king(self):
        return True

    def __str__(self):
        if self.is_white():
            return "W"
        return "B"


class Board:
    def __init__(self):  # row, col
        self.board = []
        self.white_turn = True
        self.white_fig_left = 12
        self.black_fig_left = 12
        self.black_won = False
        self.white_won = False
        self.capture_exists = False
        self.last_white_mov_indx = 0
        self.white_moves_hist = [None] * MOVES_HIST_LEN
        self.black_moves_hist = [None] * MOVES_HIST_LEN
        self.last_black_mov_indx = 0
        self.black_repeats = False
        self.white_repeats = False

        self.__set_pieces()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        result.board = deepcopy(self.board)
        return result

    def __str__(self):
        to_ret = ""
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                to_ret += str(self.board[row][col])
            to_ret += "\n"
        return to_ret

    # useful only for debugging (set board according to given list of strings)
    def set(self, b):
        self.white_fig_left = 0
        self.black_fig_left = 0
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                fig = Field()
                if b[row][col] == "b" or b[row][col] == "w":
                    fig = Pawn(b[row][col] == "w", row, col)

                if b[row][col] == "B" or b[row][col] == "W":
                    fig = King(Pawn(b[row][col] == "W", row, col))

                self.board[row][col] = fig
                if self.board[row][col].is_black():
                    self.black_fig_left += 1
                if self.board[row][col].is_white():
                    self.white_fig_left += 1

    # initializes board
    def __set_pieces(self):
        for row in range(BOARD_HEIGHT):
            self.board.append([])
            for col in range(BOARD_WIDTH):
                self.board[row].append(Field())

        for row in range(BOARD_HEIGHT // 2 - 1):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(False, row, col)

        for row in range(BOARD_HEIGHT // 2 + 1, BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                self.board[row][col] = Pawn(True, row, col)

    # get possible moves for piece
    def get_piece_moves(self, piece):
        pos_moves = []
        row = piece.row
        col = piece.col
        if piece.is_black():
            enemy_is_white = True
        else:
            enemy_is_white = False

        if piece.is_white() or (piece.is_black() and piece.is_king()):
            dir_y = -1
            if row > 0:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    # captures
                    elif (
                        self.board[new_row][new_col].is_white() == enemy_is_white
                        and new_row + dir_y >= 0
                        and new_col - 1 >= 0
                        and self.board[new_row + dir_y][new_col - 1].is_empty()
                    ):
                        pos_moves.append(
                            Move(
                                piece,
                                new_row + dir_y,
                                new_col - 1,
                                self.board[new_row][new_col],
                            )
                        )
                        self.capture_exists = True

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    # captures
                    elif (
                        self.board[new_row][new_col].is_white() == enemy_is_white
                        and new_row + dir_y >= 0
                        and new_col + 1 < BOARD_WIDTH
                        and self.board[new_row + dir_y][new_col + 1].is_empty()
                    ):
                        pos_moves.append(
                            Move(
                                piece,
                                new_row + dir_y,
                                new_col + 1,
                                self.board[new_row][new_col],
                            )
                        )
                        self.capture_exists = True

        if piece.is_black() or (piece.is_white() and self.board[row][col].is_king()):
            dir_y = 1
            if row < BOARD_WIDTH - 1:
                new_row = row + dir_y
                if col > 0:
                    new_col = col - 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    elif (
                        self.board[new_row][new_col].is_white() == enemy_is_white
                        and new_row + dir_y < BOARD_WIDTH
                        and new_col - 1 >= 0
                        and self.board[new_row + dir_y][new_col - 1].is_empty()
                    ):
                        pos_moves.append(
                            Move(
                                piece,
                                new_row + dir_y,
                                new_col - 1,
                                self.board[new_row][new_col],
                            )
                        )
                        self.capture_exists = True

                if col < BOARD_WIDTH - 1:
                    new_col = col + 1
                    if self.board[new_row][new_col].is_empty():
                        pos_moves.append(Move(piece, new_row, new_col))
                    # captures
                    elif (
                        self.board[new_row][new_col].is_white() == enemy_is_white
                        and new_row + dir_y < BOARD_WIDTH
                        and new_col + 1 < BOARD_WIDTH
                        and self.board[new_row + dir_y][new_col + 1].is_empty()
                    ):
                        pos_moves.append(
                            Move(
                                piece,
                                new_row + dir_y,
                                new_col + 1,
                                self.board[new_row][new_col],
                            )
                        )
                        self.capture_exists = True
        return pos_moves

    # get possible moves for player
    def get_possible_moves(self, is_black_turn):
        pos_moves = []
        self.capture_exists = False
        for row in range(BOARD_WIDTH):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                if not self.board[row][col].is_empty():
                    if (is_black_turn and self.board[row][col].is_black()) or (
                        not is_black_turn and self.board[row][col].is_white()
                    ):
                        pos_moves.extend(self.get_piece_moves(self.board[row][col]))
        return pos_moves

    # detect draws
    def end(self):
        # stop if repeats
        if self.black_repeats and self.white_repeats:
            # who won
            ev = basic_ev_func(self, not self.white_turn)
            if ev < 0:
                self.black_won = True
            elif ev > 0:
                self.white_won = True
            else:
                self.black_won = True
                self.white_won = True
            return True
        return False

    # used for useless play detection (game is stopped when players repeats the same moves)
    def register_move(self, move):
        move_tuple = (
            move.piece.row,
            move.piece.col,
            move.dest_row,
            move.dest_col,
            id(move.piece),
        )

        if self.white_turn:
            self.white_repeats = False
            if move_tuple in self.white_moves_hist:
                self.white_repeats = True
            self.white_moves_hist[self.last_white_mov_indx] = move_tuple
            self.last_white_mov_indx += 1
            if self.last_white_mov_indx >= MOVES_HIST_LEN:
                self.last_white_mov_indx = 0
        else:
            self.black_repeats = False
            if move_tuple in self.black_moves_hist:
                self.black_repeats = True
            self.black_moves_hist[self.last_black_mov_indx] = move_tuple
            self.last_black_mov_indx += 1
            if self.last_black_mov_indx >= MOVES_HIST_LEN:
                self.last_black_mov_indx = 0

    # execute move on board
    def make_move(self, move):
        d_row = move.dest_row
        d_col = move.dest_col
        row_from = move.piece.row
        col_from = move.piece.col

        self.board[d_row][d_col] = self.board[row_from][col_from]
        self.board[d_row][d_col].row = d_row
        self.board[d_row][d_col].col = d_col
        self.board[row_from][col_from] = Field()

        if move.captures:
            fig_to_del = move.captures
            self.board[fig_to_del.row][fig_to_del.col] = Field()
            if self.white_turn:
                self.black_fig_left -= 1
            else:
                self.white_fig_left -= 1

        if (
            self.white_turn and d_row == 0 and not self.board[d_row][d_col].is_king()
        ):  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        if (
            not self.white_turn
            and d_row == BOARD_WIDTH - 1
            and not self.board[d_row][d_col].is_king()
        ):  # damka
            self.board[d_row][d_col] = King(self.board[d_row][d_col])

        self.white_turn = not self.white_turn


class Game:
    def __init__(self, window, board):
        self.window = window
        self.board = board
        self.something_is_marked = False
        self.marked_col = None
        self.marked_row = None
        self.pos_moves = {}

    def __draw(self):
        self.window.fill(BRIGHT_BOARD_COL)
        # draw board
        for row in range(BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                y = row * FIELD_SIZE
                x = col * FIELD_SIZE
                pygame.draw.rect(
                    self.window, DARK_BOARD_COL, (x, y, FIELD_SIZE, FIELD_SIZE)
                )

        # draw pieces
        for row in range(BOARD_HEIGHT):
            for col in range((row + 1) % 2, BOARD_WIDTH, 2):
                cur_col = None
                if self.board.board[row][col].is_white():
                    cur_col = WHITE_PIECES_COL
                elif self.board.board[row][col].is_black():
                    cur_col = BLACK_PIECES_COL
                if cur_col is not None:
                    x = col * FIELD_SIZE
                    y = row * FIELD_SIZE
                    pygame.draw.circle(
                        self.window,
                        cur_col,
                        (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                        PIECE_SIZE,
                    )
                    if self.board.board[row][col].is_king():
                        pygame.draw.circle(
                            self.window,
                            KING_MARK_COL,
                            (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                            PIECE_SIZE / 2,
                        )

        # if piece is marked by user, mark it and possible moves
        if self.something_is_marked:
            x = self.marked_col * FIELD_SIZE
            y = self.marked_row * FIELD_SIZE
            pygame.draw.circle(
                self.window,
                POSS_MOVE_MARK_COL,
                (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                PIECE_SIZE + MARK_THICK,
                MARK_THICK,
            )
            pos_moves = self.board.get_piece_moves(
                self.board.board[self.marked_row][self.marked_col]
            )
            for pos_move in pos_moves:
                self.pos_moves[(pos_move.dest_row, pos_move.dest_col)] = pos_move
                x = pos_move.dest_col * FIELD_SIZE
                y = pos_move.dest_row * FIELD_SIZE
                pygame.draw.circle(
                    self.window,
                    POSS_MOVE_MARK_COL,
                    (x + FIELD_SIZE / 2, y + FIELD_SIZE / 2),
                    POS_MOVE_MARK_SIZE,
                )

    def update(self):
        self.__draw()
        pygame.display.update()

    def mouse_to_indexes(self, pos):
        return (int(pos[0] // FIELD_SIZE), int(pos[1] // FIELD_SIZE))

    def clicked_at(self, pos):
        (col, row) = self.mouse_to_indexes(pos)
        field = self.board.board[row][col]
        if self.something_is_marked:
            if (row, col) in self.pos_moves:
                self.board.make_move(self.pos_moves[(row, col)])
                self.something_is_marked = False
                self.pos_moves = {}

        if field.is_white():
            if self.something_is_marked:
                self.something_is_marked = False
                self.pos_moves = {}
            else:
                self.something_is_marked = True
                self.marked_col = col
                self.marked_row = row


def main():
    board = Board()
    window = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    is_running = True
    clock = pygame.time.Clock()
    game = Game(window, board)

    while is_running:
        clock.tick(FPS)

        if not game.board.white_turn:
            move = minimax_a_b(game.board, MINIMAX_DEPTH, True, basic_ev_func)
            # move = minimax_a_b(game.board, MINIMAX_DEPTH, True, push_forward_ev_func)
            # move = minimax_a_b(
            #     game.board, MINIMAX_DEPTH, True, push_to_opp_half_ev_func
            # )
            # move = minimax_a_b(game.board, MINIMAX_DEPTH, True, group_prize_ev_func)

            if move is not None:
                game.board.make_move(move)
            else:
                is_running = False
        if game.board.end():
            is_running = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                game.clicked_at(pos)

        game.update()

    pygame.quit()


def ai_vs_ai(ev_func, depth):
    board = Board()
    is_running = True

    while is_running:
        if board.white_turn:
            # move = minimax_a_b(board, depth, not board.white_turn, ev_func)
            move = minimax_a_b(
                board, MINIMAX_DEPTH, not board.white_turn, basic_ev_func
            )
            # move = minimax_a_b(board, 4, not board.white_turn, push_forward_ev_func)
            # move = minimax_a_b( board, 5, not board.white_turn, push_to_opp_half_ev_func)
            # move = minimax_a_b(board, 5, not board.white_turn, group_prize_ev_func)
        else:
            move = minimax_a_b(board, depth, not board.white_turn, ev_func)
            # move = minimax_a_b(board, 4, not board.white_turn, basic_ev_func)
            # move = minimax_a_b(board, 10, not board.white_turn, push_forward_ev_func)
            # move = minimax_a_b(board, 4, not board.white_turn, push_to_opp_half_ev_func)
            # move = minimax_a_b(board, 5, not board.white_turn, group_prize_ev_func)

        if move is not None:
            board.register_move(move)
            board.make_move(move)
        else:
            if board.white_turn:
                board.black_won = True
            else:
                board.white_won = True
            is_running = False
        if board.end():
            is_running = False
    print("black_won:", board.black_won)
    print("white_won:", board.white_won)
    # if both won then it is a draw!
    return board.black_won, board.white_won


# main()
# ai_vs_ai()


def ai_vs_ai_wrapper(args):
    ev_func, depth = args
    return ai_vs_ai(ev_func, depth)


def experiment():
    ev_funcs = [
        ("basic_ev_func", basic_ev_func),
        ("group_prize_ev_func", group_prize_ev_func),
        ("push_to_opp_half_ev_func", push_to_opp_half_ev_func),
        ("push_forward_ev_func", push_forward_ev_func),
    ]

    depths = range(3, 7)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for depth in depths:
            data = {
                "ev_func": [],
                "wins": [],
                "draws": [],
                "losses": [],
            }
            for ev_func_name, ev_func in ev_funcs:
                results = pool.map(
                    ai_vs_ai_wrapper, [(ev_func, depth) for _ in range(25)]
                )
                wins, draws, losses = 0, 0, 0
                for black, white in results:
                    if black and white:
                        draws += 1
                    elif black:
                        losses += 1
                    elif white:
                        wins += 1
                data["ev_func"].append(ev_func_name)
                data["wins"].append(wins)
                data["draws"].append(draws)
                data["losses"].append(losses)

            # Save results to a CSV file for the current depth
            pd.DataFrame(data).to_csv(
                f"experiment_results_depth_{depth}.csv", index=False
            )


if __name__ == "__main__":
    experiment()
    # main()
