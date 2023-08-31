import pdb
'''import sys
sys.path.append('/home/bsc98/bsc98798/.local/lib/python3.10/site-packages/')
from chess import *'''
import itertools
import chess
from Chess import actions, de_action, FEN, Fen_df, fer_df, turn, eval_discart, full_tree, sarsa_actions, gen_number
from SARSA_Chess import jugades_possibles, classify_board
import numpy
import random
import datetime

before = str(datetime.datetime.now().time()).split(':')
jug_poss = jugades_possibles()

def print_time():
    tot_time = [' hours,', ' minutes,', ' seconds.']
    after = str(datetime.datetime.now().time()).split(':')
    before_s = int(before[-3]) * 60 * 60 + int(before[-2]) * 60 + float(before[-1])
    after_s = int(after[-3]) * 60 * 60 + int(after[-2]) * 60 + float(after[-1])
    if after_s > before_s:
        t_diff = after_s - before_s
        if t_diff < 60:
            tot_time[-3] = '0' + tot_time[-3]
            tot_time[-2] = '0' + tot_time[-2]
            tot_time[-1] = str(t_diff) + tot_time[-1]
        elif t_diff < (60 * 60):
            tot_time[-3] = '0' + tot_time[-3]
            tot_time[-2] = str(int(t_diff // 60)) + tot_time[-2]
            tot_time[-1] = str(t_diff % 60) + tot_time[-1]
        elif t_diff < (60 * 60 * 24):
            tot_time[-3] = str(int(t_diff // 3600)) + tot_time[-3]
            t_diff = t_diff % 3600
            tot_time[-2] = str(int(t_diff // 60)) + tot_time[-2]
            tot_time[-1] = str(t_diff % 60) + tot_time[-1]
    print(tot_time[0], tot_time[1], tot_time[2])


def move_choice(act, s_inicial, alpha, gamma, jug_poss, q_scoring, mov_num):
    def u_func(state_a, accions):
        b_1 = chess.Board(FEN(state_a))
        torn = turn(b_1.epd())
        best_q = -1234567890
        best_a = random.choice(accions)
        best_s = state_a
        for a_u in accions:
            b = chess.Board(FEN(state_a))
            movi = de_action(a_u, b)
            b.push(chess.Move.from_uci(movi))  # b.push(movi)
            st = Fen_df(b.epd())
            # q = eval_discart(st)
            clas_bo = classify_board(best_s, mov_num)
            q = q_scoring[clas_bo[0]][clas_bo[1]][clas_bo[2]][act.index(a_u)]
            if type(q) == str:
                continue
            if q > best_q:
                best_q = q
                best_a = a_u
                best_s = st
        return [best_q, best_a, best_s]
    def r_func(state_b, accions):
        b_1 = chess.Board(FEN(state_b))
        torn_r = turn(b_1.epd())
        best_q = -1234567890
        best_a = -1234567890
        best_s = state_b
        for a_r in accions:
            b = chess.Board(FEN(state_b))
            movi = de_action(a_r, b)
            b.push(chess.Move.from_uci(movi))
            st = Fen_df(b.epd())
            if torn_r:
                q = eval_discart(st)
            else:
                q = eval_discart(st) * -1
            if q > best_q:
                best_q = q
                best_a = a_r
                best_s = st
        return [best_q, best_a, best_s]
    # Choose move:
    best = u_func(s_inicial, jug_poss)
    board_1 = chess.Board(FEN(best[-1]))
    # Reward
    a_2 = sarsa_actions([board_1])[0]
    reward = r_func(best[-1], a_2)
    reward[0] = best[0] - reward[0]
    # Update State-Action table:
    cls_bo = classify_board(best[-1], mov_num)
    action_index = act.index(best[1])
    current_score = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][action_index]
    q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][action_index] = current_score + alpha * (reward[0] + gamma * best[0] - current_score)
    return best


while 1:
    doc = open('Guardar_data2', 'r+')
    doc = doc.read().split(',')
    doc = [doc[:len(doc) // 2], doc[len(doc) // 2:]]
    for d in range(len(doc)):
        doc[d] = [doc[d][:len(doc[d]) // 3], doc[d][len(doc[d]) // 3:len(doc[d]) * 2 // 3],
                  doc[d][len(doc[d]) * 2 // 3:]]
    for d1 in range(2):
        for d2 in range(3):
            doc[d1][d2] = [doc[d1][d2][:len(doc[d1][d2]) // 3],
                           doc[d1][d2][len(doc[d1][d2]) // 3:len(doc[d1][d2]) * 2 // 3],
                           doc[d1][d2][len(doc[d1][d2]) * 2 // 3:]]
    for d3 in range(2):
        for d4 in range(3):
            for d5 in range(3):
                for d6 in range(len(doc[d3][d4][d5])):
                    if not doc[d3][d4][d5][d6] == '':
                        doc[d3][d4][d5][d6] = float(doc[d3][d4][d5][d6])
    q_scores_split_training = doc
    color = input(' Tria el teu color o cancel·la la partida: (blanques(b), negres(n), cancel·la(c))   \n -->  ').upper()
    if color == 'C':
        print(' Partida cancel·lada')
        break
    elif color == 'B':
        board = chess.Board()
        exiting = False
        move = 0
        while (not board.is_game_over()) and (not exiting):
            print()
            print(board)
            print()
            jugs_poss = []
            for le_mo in list(board.legal_moves):
                jugs_poss.append(str(le_mo))
            if not(move % 2):
                mov = input(' Introdueix el moviment: \n -->  ')
                if mov not in jugs_poss:
                    exiting = input(' El moviment no es pot efectuar, vol abandonar la partida? (si(y), no(n)) \n -->  ').upper() == 'Y'
                    if exiting:
                        break
                    elif not exiting:
                        continue
            else:
                jugs_poss = sarsa_actions([board])[0]
                # mov = random.choice(jug_poss)
                mov = move_choice(jug_poss, Fen_df(board.epd()), 0.3, 0.6, jugs_poss, q_scores_split_training, move)
                mov = de_action(mov[1], board)
            board.push_uci(mov)
            if board.is_game_over():
                if board.is_checkmate():
                    if move % 2:
                        input("T'han fet escac i mat, ja guanyaràs en una altra ocasió (Prem enter per a continuar) \n -->  ")
                    else:
                        input("Has fet escac i mat, enhorabona. (Prem enter per a continuar) \n -->  ")
            move += 1
    elif color == 'N':
        board = chess.Board()
        exiting = False
        move = 0
        while (not board.is_game_over()) and (not exiting):
            print()
            print(board)
            print()
            jugs_poss = []
            for le_mo in list(board.legal_moves):
                jugs_poss.append(str(le_mo))
            if move % 2:
                mov = input(' Introdueix el moviment: \n -->  ')
                if mov not in jugs_poss:
                    exiting = input(
                        ' El moviment no es pot efectuar, vol abandonar la partida? (si(y), no(n)) \n -->  ').upper() == 'Y'
                    if exiting:
                        break
                    elif not exiting:
                        continue
            else:
                jugs_poss = sarsa_actions([board])[0]
                # mov = random.choice(jug_poss)
                mov = move_choice(jug_poss, Fen_df(board.epd()), 0.3, 0.6, jugs_poss, q_scores_split_training, move)
                mov = de_action(mov[1], board)
            board.push_uci(mov)
            if board.is_game_over():
                if board.is_checkmate():
                    if not(move % 2):
                        input("T'han fet escac i mat, ja guanyaràs en una altra ocasió (Prem enter per a continuar) \n -->  ")
                    else:
                        input("Has fet escac i mat, enhorabona. (Prem enter per a continuar) \n -->  ")
            move += 1


# TODO implementar el SARSA_Chess
