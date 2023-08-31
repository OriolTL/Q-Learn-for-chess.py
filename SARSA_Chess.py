import pdb
'''import sys
sys.path.append('/home/bsc98/bsc98798/.local/lib/python3.10/site-packages/')
from chess import *'''
import itertools
import chess
from Chess import actions, de_action, FEN, Fen_df, fer_df, turn, eval_discart, full_tree, sarsa_actions, gen_number
import numpy
import random
import datetime

as_ps = []

before = str(datetime.datetime.now().time()).split(':')
# print(actions([chess.Board()]))
# print(Fen_df(chess.Board().epd()))
# print(eval_discart(Fen_df(chess.Board().epd())))


def print_time():
    tot_time = [0, 0, 0]
    after = str(datetime.datetime.now().time()).split(':')
    before_s = int(before[-3]) * 60 * 60 + int(before[-2]) * 60 + float(before[-1])
    after_s = int(after[-3]) * 60 * 60 + int(after[-2]) * 60 + float(after[-1])
    if after_s > before_s:
        t_diff = after_s - before_s
        if t_diff < 60:
            tot_time[-3] = 0
            tot_time[-2] = 0
            tot_time[-1] = t_diff
        elif t_diff < (60 * 60):
            tot_time[-3] = 0
            tot_time[-2] = t_diff // 60
            tot_time[-1] = t_diff % 60
        elif t_diff < (60 * 60 * 24):
            tot_time[-3] = t_diff // 3600
            t_diff = t_diff % 3600
            tot_time[-2] = t_diff // 60
            tot_time[-1] = t_diff % 60
    return tot_time


def jugades_possibles():
    j_p = []
    for h in range(8):
        h += 1
        for h1 in range(8):
            h1 += 1
            for h2 in range(8):
                h2 += 1
                for h3 in range(8):
                    h3 += 1
                    for p in ['p', 'n', 'b', 'r', 'q', 'k', 'K', 'Q', 'R', 'B', 'N', 'P']:
                        j_p.append([h, h1, h2, h3, p])
    return j_p


def estats_possibles():
    dat_fs = []
    if True:
        pieces = []
        pic = list(range(-5, 6))
        pic.remove(0)
        while pieces.count(-1) < 8:
            pieces.append(-1)
        while pieces.count(1) < 8:
            pieces.append(1)
        while pieces.count(-2) < 2:
            pieces.append(-2)
        while pieces.count(2) < 2:
            pieces.append(2)
        while pieces.count(3) < 2:
            pieces.append(3)
        while pieces.count(-3) < 2:
            pieces.append(-3)
        while pieces.count(4) < 2:
            pieces.append(4)
        while pieces.count(-4) < 2:
            pieces.append(-4)
        while pieces.count(5) < 1:
            pieces.append(5)
        while pieces.count(-5) < 1:
            pieces.append(-5)
    for v in range(32, 63):
        print(v)
        dat_f = []
        for e in range(v):
            dat_f.append(0)
        dat_f.extend([-6, 6])
        a_left = 64 - len(dat_f)
        if a_left > 0:
            comb = itertools.combinations(pieces, a_left)
            ext_vals = list(comb)
            for ext_val in ext_vals:
                while ext_vals.count(ext_val) > 1:
                    ext_vals.remove(ext_val)
            for ext_val in ext_vals:
                n_d_f = dat_f.copy()
                n_d_f.extend(ext_val)
                dat_fs.append(n_d_f)
            pdb.set_trace()
        else:
            dat_fs.append(dat_f)
    pdb.set_trace()
# estats_possibles() NO es pot calcular tots els estats, n'hi ha prop de 4*10^44 (la terra te 10^50 atoms per comparar)


def classify_board(df, move):
    bord = chess.Board(FEN(df))
    # Closed or open game
    if (df[:64].count(1) >= 6) and (df[:64].count(-1) >= 6):
        # Closed game
        bo_class = [0]
    else:
        # Open game
        bo_class = [1]
    # King-sided, Queen-sided or Non-defined game
    if True:
        e4 = str(bord.piece_at(28))
        d4 = str(bord.piece_at(27))
        e5 = str(bord.piece_at(36))
        d5 = str(bord.piece_at(35))
    if e4 == 'P':
        if d4 != 'P':
            if e5 == 'p':
                # King-side game
                bo_class.append(0)
            else:
                # Non-defined game
                bo_class.append(2)
        elif d4 == 'P':
            # Non-defined game
            bo_class.append(2)
    elif d4 == 'P':
        if e4 != 'P':
            if d5 == 'p':
                # Queen-side game
                bo_class.append(1)
            else:
                # Non-defined game
                bo_class.append(2)
        elif e4 == 'P':
            # Non-defined game
            bo_class.append(2)
    else:
        # Non-defined game
        bo_class.append(2)
    # Opening, Middle-game or Endgame
    if df[:64].count(0) > 48:
        # Endgame
        bo_class.append(2)
    elif move < 14:
        # Opening
        bo_class.append(0)
    else:
        # Middle-game
        bo_class.append(1)
    return bo_class
# Classificar els estats de manera que es poden agrupar per tipus i millorar la qualitat de les jugades


def qlearn(act, rewards, s_inicial, alpha, gamma, max_iters, q_scoring=None):
    ''' act and rewards are lists, alpha and gamma have to be numbers (float probably) '''
    # Fitness function//Utility function:
    # TODO aquesta funcio ha de fer servir la q_scoring per a poder fer recursivitat i canviar els resultats.
    def u_func(state_a, accions):
        # print('Inicial:')
        # print(chess.Board(FEN(state_a)))
        # print()
        b_1 = chess.Board(FEN(state_a))
        torn = turn(b_1.epd())
        best_q = -1234567890
        best_a = random.choice(accions)
        best_s = state_a
        '''
        cls_tb = classify_board(best_s, iter_count)
        if list(q_scoring[cls_tb[0]][cls_tb[1]][cls_tb[2]]).count(0) > 0:
            un_acts = []
            t_b = chess.Board(FEN(best_s))
            for un_act in accions:
                if q_scoring[cls_tb[0]][cls_tb[1]][cls_tb[2]][act.index(best_a)] == 0:
                    un_acts.append(un_act)
                    best_a = random.choice(un_acts)
            t_b.push(chess.Move.from_uci(de_action(best_a, chess.Board(FEN(best_s)))))
            cls_bo = classify_board(best_s, iter_count)
            return [q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(best_a)], best_a, Fen_df(t_b.epd())]
        '''
        for a_u in accions:
            b = chess.Board(FEN(state_a))
            mov = de_action(a_u, b)
            mov = chess.Move.from_uci(mov)
            b.push(mov)
            # print(b)
            # print()
            st = Fen_df(b.epd())
            # q = eval_discart(st)
            cls_bo = classify_board(best_s, iter_count)
            q = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(a_u)]
            if type(q) == str:
                continue
            if q > best_q:
                best_q = q
                best_a = a_u
                best_s = st
        return [best_q, best_a, best_s]
    # t_u = u_func(s_inicial, valid_act)
    # print(t_u)
    # print(chess.Board(FEN(t_u[2])))
        # ------------------------------
    # Reward: TO DO

    def r_func(state_b, accions):
        # print('Inicial:')
        # print(chess.Board(FEN(state_b)))
        # print()
        b_1 = chess.Board(FEN(state_b))
        torn_r = turn(b_1.epd())
        best_q = -1234567890
        best_a = -1234567890
        best_s = state_b
        for a_r in accions:
            b = chess.Board(FEN(state_b))
            mov = de_action(a_r, b)
            mov = chess.Move.from_uci(mov)
            b.push(mov)
            # print(b)
            # print()
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
    # t_b = chess.Board(FEN(t_u[2]))
    # print(chess.Board(FEN(r_func(t_u[2], actions([t_b])[0])[2])))
        # ------------------------------
    # Inicialitza scoring, actions & states:
    ''' Fer v_func per a diferenciar totes les jugades '''

    if q_scoring is None:
        # q_scoring = numpy.zeros([len(rewards[0]), len(rewards[1]), len(act)])
        q_scoring = [[[numpy.zeros([len(act)]), numpy.zeros([len(act)]), numpy.zeros([len(act)])], [numpy.zeros([len(act)]), numpy.zeros([len(act)]), numpy.zeros([len(act)])], [numpy.zeros([len(act)]), numpy.zeros([len(act)]), numpy.zeros([len(act)])]],[[numpy.zeros([len(act)]), numpy.zeros([len(act)]), numpy.zeros([len(act)])], [numpy.zeros([len(act)]), numpy.zeros([len(act)]), numpy.zeros([len(act)])], [numpy.zeros([len(act)]), numpy.zeros([len(act)]), numpy.zeros([len(act)])]]]
    s = s_inicial
    ''' a ha de ser nomes les accions valides '''
    a = sarsa_actions([chess.Board(FEN(s))])[0]
    # Inicialitza loop-breaking values: TODO
    iter_count = 0
    a_s_b = []
    three_f_r = False
    jug_fet = []
    # Loop:
    while iter_count < max_iters:
        # Get max Q in status:
        best = u_func(s, a)
        # print(chess.Board(FEN(best[-1])))

        # is three-fold repetition?
        board_1 = chess.Board(FEN(best[-1]))
        if a_s_b.count(board_1) == 3:
            cls_bo = classify_board(best[-1], iter_count)
            c_sc = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(best[1])]
            if c_sc == '':
                c_sc = - 10
            q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(best[1])] = -10 * abs(c_sc)
            break
        else:
            a_s_b.append(board_1)

        # Game over: TODO recompensa
        if board_1.is_game_over():
            if board_1.is_checkmate():
                print('Checkmate')
                print(board_1)
                cls_bo = classify_board(best[-1], iter_count)
                c_sc = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(best[1])]
                q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(best[1])] = 10 * abs(c_sc)
                for acte in range(len(jug_fet), 0):
                    if (acte % 2) == (jug_fet.index(jug_fet[-1])):
                        c_scr = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(jug_fet[acte])]
                        q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(jug_fet[acte])] = (acte / len(jug_fet)) * c_scr
                        if q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(jug_fet[acte])]:
                            break
                    else:
                        c_scr = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(jug_fet[acte])]
                        q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(jug_fet[acte])] = -(acte / (len(jug_fet) * 2)) * c_scr
                        if q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][act.index(jug_fet[acte])]:
                            break
            break

        # Solve reward: TODO (improve reward system)
        a_2 = sarsa_actions([board_1])[0]
        reward = r_func(best[-1], a_2)
        # print(chess.Board(FEN(reward[-1])))
        reward[0] = best[0] - reward[0]

        # Update State-Action table:
        cls_bo = classify_board(best[-1], iter_count)
        action_index = act.index(best[1])
        current_score = q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][action_index]
        if type(current_score) == float:
            q_scoring[cls_bo[0]][cls_bo[1]][cls_bo[2]][action_index] = current_score + alpha * (reward[0] + gamma * best[0] - current_score)

        # Change status:
        jug_fet.append(best[1])
        prev_status = s
        s = best[-1]
        a = sarsa_actions([chess.Board(FEN(s))])[0]
        iter_count += 1
        print('Iteration:       ', iter_count)
        print('                 Previous position')
        print(chess.Board(FEN(prev_status)))
        print('                 Best position    ')
        print(chess.Board(FEN(best[-1])))
        print('Best action      ', de_action(best[1], chess.Board(FEN(best[-1]))))
        print('Best value       ', best[0])
        print('Reward is        ', reward[0])
        print('Fen string is:   ', str(chess.Board(FEN(best[-1])).epd()))
        print()
    return q_scoring


def main():
    #max_train_t = 600
    max_train_t = [int(input('Hores: ')), int(input('Minuts: ')), int(input('Segons: '))]
    train_t = 0
    jugs_possib = jugades_possibles()
    doc = open('Guardar_data2', 'r+')
    doc = doc.read().split(',')
    if '[' not in doc:
        doc = [doc[:len(doc)//2], doc[len(doc)//2:]]
        for d in range(len(doc)):
            doc[d] = [doc[d][:len(doc[d])//3], doc[d][len(doc[d])//3:len(doc[d])*2//3], doc[d][len(doc[d])*2//3:]]
        for d1 in range(2):
            for d2 in range(3):
                doc[d1][d2] = [doc[d1][d2][:len(doc[d1][d2])//3], doc[d1][d2][len(doc[d1][d2])//3:len(doc[d1][d2])*2//3], doc[d1][d2][len(doc[d1][d2])*2//3:]]
        for d3 in range(2):
            for d4 in range(3):
                for d5 in range(3):
                    for d6 in range(len(doc[d3][d4][d5])):
                        if not doc[d3][d4][d5][d6] == '':
                            doc[d3][d4][d5][d6] = float(doc[d3][d4][d5][d6])
        q_scores_split_training = doc
    else:
        q_scores_split_training = None
    q_sc = (qlearn(jugs_possib, numpy.zeros([8, 8], dtype=float),
                   Fen_df(chess.Board().epd()), 0.3, 0.6, 100, q_scores_split_training))
    while True:
        t_transcurred = print_time()
        if (t_transcurred[0] * 3600 + t_transcurred[1] * 60 + t_transcurred[2]) > (max_train_t[0] * 3600 + max_train_t[1] * 60 + max_train_t[2]):
            break
        q_sc = (qlearn(jugs_possib, numpy.zeros([8, 8], dtype=float), Fen_df(chess.Board().epd()), 0.3, 0.6, 100, q_sc))
        train_t += 1
        print('------------------------------------------------------------------', train_t)
    doc2 = open('Guardar_data2', 'r+')
    doc2.truncate(0)
    for h in q_sc:
        for h1 in h:
            for h2 in h1:
                for h3 in h2:
                    doc2.write(str(h3) + ',')
    doc2.close()


if __name__ == '__main__':
    main()
print(print_time())
