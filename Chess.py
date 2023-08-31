import pdb
import random
import datetime

import chess
import pandas as pd
from Chess_Model_0 import TAULELL, df_taulell, aval_pos, re_taulell, title_df
from numpy import array

before = str(datetime.datetime.now().time()).split(':')
# data_file = open('Guardar_data.docx','r+')
board = chess.Board()
dict_1 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
dict_2 = {'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6, 'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
          'None': 0}
dict_3 = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h'}
dict_v = {0: 0, 1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0, -1: -1, -2: -3, -3: -3, -4: -5, -5: -9, -6: 0}
dict_v_2 = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 0, 'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': 0}
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
          109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
          179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293,
          307, 311]
calc_boards = []
# 517387461950094663192259619090856391796537069755726894404269260800000000000000000000000000000000001,0.2

# ll_m = list(taulell.legal_moves)
# print(ll_m)
# print(random.choice(ll_m))
# taulell.push(random.choice(ll_m))

'''
fen = board.epd()
print(fen)
t = TAULELL(fen)
#print(array(t))
df_t = df_taulell(t)
print(df_t)
'''
# data_file.write('Funciona')
# print((data_file.readline(-1)))


def print_time():
    tot_time = [' hours,', ' minutes,', ' seconds.']
    after = str(datetime.datetime.now().time()).split(':')
    if not(after[0] == before[0]):
        tot_time[0] = str(int(after[0]) - int(before[0])) + tot_time[0]
    else:
        tot_time[0] = '0' + tot_time[0]
    if not(after[1] == before[1]):
        tot_time[1] = str(int(after[1]) - int(before[1])) + tot_time[1]
    else:
        tot_time[1] = '0' + tot_time[1]
    if not(after[2] == before[2]):
        tot_time[2] = str(float(after[2]) - float(before[2])) + tot_time[2]
    else:
        tot_time[2] = '0' + tot_time[2]
    print(tot_time[0], tot_time[1], tot_time[2])


def turn(fen):
    l_fen = fen.split(' ')
    if 'w' in l_fen:
        return 1
    if 'b' in l_fen:
        return 0


def FEN(df):
    FEN_str = ""
    taulell = []
    board = df[:64]
    other = df[64:]
    for i in range(0, len(board), 8):
        taulell.append(board[i:i + 8])
    for x in range(8):
        fila = taulell[x]
        empty = 0
        for y in range(8):
            casella = fila[y]
            if casella == 0:
                empty += 1
            elif (casella != 0) and (empty > 0):
                FEN_str = FEN_str + str(empty)
                empty = 0
            if casella == -1:
                FEN_str = FEN_str + "p"
            elif casella == -2:
                FEN_str = FEN_str + "n"
            elif casella == -3:
                FEN_str = FEN_str + "b"
            elif casella == -4:
                FEN_str = FEN_str + "r"
            elif casella == -5:
                FEN_str = FEN_str + "q"
            elif casella == -6:
                FEN_str = FEN_str + "k"
            elif casella == 1:
                FEN_str = FEN_str + "P"
            elif casella == 2:
                FEN_str = FEN_str + "N"
            elif casella == 3:
                FEN_str = FEN_str + "B"
            elif casella == 4:
                FEN_str = FEN_str + "R"
            elif casella == 5:
                FEN_str = FEN_str + "Q"
            elif casella == 6:
                FEN_str = FEN_str + "K"
            if (y == 7) and (empty != 0):
                FEN_str = FEN_str + str(empty)
        if x != 7:
            FEN_str = FEN_str + "/"
    if other[0] == 1:
        FEN_str = FEN_str + " w "
    else:
        FEN_str = FEN_str + " b "
    if other[1] == 1:
        FEN_str = FEN_str + "K"
    if other[2] == 1:
        FEN_str = FEN_str + "Q"
    if other[3] == 1:
        FEN_str = FEN_str + "k"
    if other[4] == 1:
        FEN_str = FEN_str + "q"
    return FEN_str


def fer_df(taulells):
    t_dfs = []
    for t in range(len(taulells)):
        if type(taulells[t]) != str:
            fen = taulells[t].epd()
        else:
            fen = taulells[t]
        df_64 = df_taulell(TAULELL(fen))
        # print(df_64)
        df_64.append(turn(fen))
        if chess.Board.has_kingside_castling_rights(taulells[t], True):
            df_64.append(1)
        else:
            df_64.append(0)
        if chess.Board.has_queenside_castling_rights(taulells[t], True):
            df_64.append(1)
        else:
            df_64.append(0)
        if chess.Board.has_kingside_castling_rights(taulells[t], False):
            df_64.append(1)
        else:
            df_64.append(0)
        if chess.Board.has_queenside_castling_rights(taulells[t], False):
            df_64.append(1)
        else:
            df_64.append(0)
        if taulells[t].is_check() and (df_64[64] == 1):
            df_64.append(1)
        else:
            df_64.append(0)
        if taulells[t].is_check() and (df_64[64] == 0):
            df_64.append(1)
        else:
            df_64.append(0)
        # print(df_64)
        t_dfs.append(df_64)
    return t_dfs

    # GENERAR DATA


def Fen_df(fen):
    df = []
    fen_split = fen.split(' ')
    fen_files = fen_split[0].split('/')
    other = fen_split[1:]
    for f in fen_files:
        squares = list(f)
        for s in squares:
            if s in dict_2:
                df.append(dict_2[s])
            else:
                for times in range(int(s)):
                    df.append(0)
    if other[0] == 'w':
        df.append(1)
    else:
        df.append(0)
    l_cast = list(other[1])
    if 'K' in l_cast:
        df.append(1)
    else:
        df.append(0)
    if 'Q' in l_cast:
        df.append(1)
    else:
        df.append(0)
    if 'q' in l_cast:
        df.append(1)
    else:
        df.append(0)
    if 'k' in l_cast:
        df.append(1)
    else:
        df.append(0)
    if len(chess.Board(fen).attackers(False, int(chess.Board(fen).king(True)))) != 0:
        df.append(1)
    else:
        df.append(0)
    if len(chess.Board(fen).attackers(True, int(chess.Board(fen).king(False)))) != 0:
        df.append(1)
    else:
        df.append(0)
    return df


'''
i = int(input("Jugades en total: "))
df_ts = []
while i != 0:
    i -= 1

    fen = board.epd()
    t = TAULELL(fen)
    df_t = df_taulell(t)
    df_t.append(1)
    if not board.is_check():
        df_t.append(0)
    else:
        df_t.append(-1)
    ta = re_taulell(t, {0: '--', -1: 'PN', -2: 'CN',
                        -3: 'AN', -4: 'TN', -5: 'DN', -6: 'RN', 1: 'PB', 2: 'CB', 3: 'AB', 4: 'TB', 5: 'DB', 6: 'RB'})
    mov_b = ESCACS_23_05_2023_v2.moviments_b(ta,[])
    mov_n = ESCACS_23_05_2023_v2.moviments_n(ta, [])
    rei_b = 0
    rei_n = 0
    for m in mov_b:
        if m[0] == 'RB':
            rei_b += 1
    for m in mov_n:
        if m[0] == 'RN':
            rei_n += 1
    df_t.extend([rei_b,rei_n])

    df_ts.append(df_t)
    print(array(ta))
    print('-------------------------------------------------------------------',aval_pos([df_t]))
    print('------------------------------------------------------------------------')

    llista_mov = list(board.legal_moves)
    board.push(random.choice(llista_mov))

    fen = board.epd()
    t = TAULELL(fen)
    df_t = df_taulell(t)
    df_t.append(0)
    if not board.is_check():
        df_t.append(0)
    else:
        df_t.append(1)
    ta =re_taulell(t, {0: '--', -1: 'PN', -2: 'CN',
                         -3: 'AN', -4: 'TN', -5: 'DN', -6: 'RN', 1: 'PB', 2: 'CB', 3: 'AB', 4: 'TB', 5: 'DB', 6: 'RB'})
    mov_b = ESCACS_23_05_2023_v2.moviments_b(ta, [])
    mov_n = ESCACS_23_05_2023_v2.moviments_n(ta, [])
    rei_b = 0
    rei_n = 0
    for m in mov_b:
        if m[0] == 'RB':
            rei_b += 1
    for m in mov_n:
        if m[0] == 'RN':
            rei_n += 1
    df_t.extend([rei_b,rei_n])

    df_ts.append(df_t)
    print(array(ta))
    print('-------------------------------------------------------------------',aval_pos([df_t]))
    print('------------------------------------------------------------------------')

    llista_mov = list(board.legal_moves)
    board.push(random.choice(llista_mov))
'''


# FONDARIA
def fondo(taulells):
    taulells_b = []
    for times in range(len(taulells)):
        taulell = taulells[times]
        if type(taulell) == list:
            taulell = chess.Board(FEN(taulell))
        accions = list(taulell.legal_moves)
        # pdb.set_trace()
        for a in range(len(accions)):
            t = taulell.copy()
            move = str(accions[a])
            t.push_uci(move)
            taulells_b.extend([t])
            if t.is_checkmate():
                # print(" MATE! ")
                pass
            if t.is_stalemate():
                # print(" STALEMATE.. ")
                pass
    return taulells_b


def long(taulells, width):
    taulells_b = []
    for times in range(len(taulells)):
        taulell = taulells[times]
        if type(taulell) == list:
            taulell = chess.Board(FEN(taulell))
        accions = list(taulell.legal_moves)
        for a in range(width):
            t = taulell.copy()
            if len(accions) != 0:
                tria = random.choice(accions)
            else:
                if taulell.is_checkmate():
                    # print(" MATE! ")
                    pass
                if taulell.is_stalemate():
                    # print(" STALEMATE.. ")
                    pass
                continue
            move = str(tria)
            t.push_uci(move)
            taulells_b.extend([t])
    return taulells_b


# Full tree:
# Approach 1
'''
taulells = chess.Board()
fond_1 = fondo([taulells])
print(len(fond_1))
fond_2 = fondo(fond_1)
print(len(fond_2))
fond_3 = fondo(fond_2)
print(len(fond_3))
fond_4 = fondo(fond_3)
print(len(fond_4))
fond_5 = fondo(fond_4)
print(len(fond_5))
'''
# Approach 2
'''
taulells = [chess.Board()]
for depth in range(int(input("Depth:   "))):
    taulells = fondo(taulells)
#print(len(taulells))
'''


def full_tree(taulells, depth):
    for depth in range(depth):
        taulells = fondo(taulells)
    return taulells


# print(len(full_tree([chess.Board()],2)))

# Random Line:
'''
taulell = chess.Board()
for depth in range(int(input("Depth  "))):
    taulell.push(random.choice(list(taulell.legal_moves)))
    print(array(taulell))
'''


def r_line(taulell, choices):
    ts = []
    for depth in range(len(choices)):
        taulell.push(list(taulell.legal_moves)[choices[depth]])
        ts.append(taulell)
    return ts


# print(r_line(chess.Board(),[13,10,0]))

# Fitted width:
'''
taulell = [chess.Board()]
for depth in range(int(input("Depth  "))):
    taulell = long(taulell,int(input("Width  ")))
    #print(array(taulell))
    print(len(taulell))
'''


def fit_width(taulell, widths):
    taulells = []
    for depth in range(len(widths)):
        taulell = long(taulell, widths[depth])
        # print(array(taulell))
        # print(len(taulell))
        taulells.extend(taulell)
    return taulells


# print(len(fit_width([chess.Board()],[20,20,3,3,3])))

# CREAR ESTATS:
'''
taulells = fit_width([chess.Board()],[20,20,4,3,2])
np_features = fer_df(taulells)
titols = (title_df[:64])
titols.extend(['Turn','w hKcr','w hQcr','b hKcr','b hQcr','w is.c','b is.c'])
df_t = pd.DataFrame(np_features,columns= titols)
print(df_total)
#print(df_total.shape)
'''


# CREAR ACCIONS:
def actions(board_types):
    act = []
    for t in range(len(board_types)):
        l_m = list(board_types[t].legal_moves)
        act.append(l_m)
    accions = []
    for moves in act:
        m = 0
        for move in moves:
            a = list(str(move))
            c = 0
            for co in a:
                if co in dict_1:
                    a[c] = dict_1[co]
                else:
                    a[c] = int(co)
                c += 1
            moves[m] = a
            m += 1
        accions.append(moves)
    return accions


def de_action(action, board_type):
    c = 0
    move = ''
    for co in action[:4]:
        if (c % 2) != 0:
            move = move + str(co)
        else:
            move = move + str(dict_3[co])
        c += 1
    if ((str(board_type.piece_at(action[0] * action[1] + (8 - action[0]) * (action[1] - 1) - 1)) == 'p') or (str(board_type.piece_at(action[0] * action[1] + (8 - action[0]) * (action[1] - 1) - 1)) == 'P')) and ((action[-2] == 8) or (action[-2] == 1)):
        move = move + action[-1]
    return move


def sarsa_actions(board_types):
    act = []
    for t in range(len(board_types)):
        l_m = list(board_types[t].legal_moves)
        act.append(l_m)
    accions = []
    for mo in range(len(act)):
        moves = act[mo]
        m = 0
        sip = False
        for move in moves:
            ac = list(str(move))
            si = False
            if len(ac) == 5:
                pieza = ac[-1]
                ac = ac[:4]
                sip = True
                si = True
                #pdb.set_trace()
            c = 0
            for co in ac:
                if co in dict_1:
                    ac[c] = dict_1[co]
                else:
                    ac[c] = int(co)
                c += 1
            if si:
                ac.append(pieza)
            else:
                ac.append(str(board_types[mo].piece_at(ac[0] * ac[1] + (8 - ac[0]) * (ac[1] - 1) - 1)))
            moves[m] = ac
            m += 1
        accions.append(moves)
    return accions
# print(sarsa_actions([chess.Board()])[0])
# TODO

# str_move = de_action(actions([chess.Board()])[0][0])

# print(actions([chess.Board()]))
'''
taulells = fit_width([chess.Board()],[20,20,4,3,2])
act_taulells = actions(taulells)
'''
# Es pot triar taulell (estat) i les accions que li corresponen:
'''
while 1:
    n = random.randint(0,len(action_total))
    print(taulells[n])
    print(actions(taulells)[n])
'''

# REWARD SYSTEM:
# IDEES
''' 
Every move's value in the match will be the addittion of If_checkmate values, If_capture values and Else values. 

n = number of moves (counting moves as rounds, one white move and one black move)
- If checkmate the chosen actions's rewards will be:
    The "n"th move will grant the winning side n-1/n points, the "n-1"th move will grant 
    the winning side (n-2)/n ponits and so on until n == 1 and therefore the points awarded are 0.
    For the losing side the "n"th move will grant them -(n-1/2n) points, the "n-1"th move
    will award -(n-2/3n) until the reward is equal to 0.
- If a piece is captured:
    v = value of the captured piece; k = value of the capturing piece; 8 is the maximum difference of value;
    The capturing side will be rewarded: if (v-k)>=0: (v-k), else: abs(v-k)/8
    The captured side will be awarded the capturing's side value times -1 on the move n, and on the "n-1"th
    move the number of points will be a tenth of the previous.
- Else:
    The reward given for random moves will be a random number from -0.25 to 0.25

'''


# EVALUACIo PRIMERA: TO DO  completar eval_discart()
def k_safe(board, square):
    col = chess.square_file(square)
    fila = chess.square_rank(square)
    e_color = 'k' == str(board.piece_at(square))
    f_color = 'K' == str(board.piece_at(square))
    safety = 0
    if fila == 0:
        vertical = [0, 1]
    elif fila == 7:
        vertical = [-1, 0]
    else:
        vertical = [-1, 0, 1]
    if col == 0:
        horitzontal = [0, 1]
    elif fila == 7:
        horitzontal = [-1, 0]
    else:
        horitzontal = [-1, 0, 1]
    for y in vertical:
        for x in horitzontal:
            e_att_p = list(board.attackers(e_color, chess.square(col + y, fila + x)))
            f_att_p = list(board.attackers(f_color, chess.square(col + y, fila + x)))
            e_att = - len(e_att_p)
            f_att = len(f_att_p)
            safety += (e_att + f_att) / ((sum(e_att_p) * sum(f_att_p)) + 1)
            return safety


def knight_outpost(board, square):
    # col = chess.square_file(square)
    # fila = chess.square_rank(square)
    # TO DO: veure si el suporta un peo
    if (board.piece_at(square) == 'N') and (board.piece_at(square + 8) == 'p'):
        return True
    elif (board.piece_at(square) == 'n') and (board.piece_at(square + 8) == 'P'):
        return True
    else:
        return False


def pawnchain(board, square):
    width = 0
    col = chess.square_file(square)
    fila = chess.square_rank(square)
    if (fila == 7) or (fila == 0):
        return [width, (square % 2)]
    if col > 0:
        if (board.piece_at(square)) == (board.piece_at(square + 7)):
            width += 1
    if col < 7:
        if (board.piece_at(square)) == (board.piece_at(square + 9)):
            width += 1
    if (1 < col < 6) and (2 < fila < 5):
        width *= 1.5
    if (2 < col < 5) and (2 < fila < 5):
        width *= 1.5
    return [width, (square % 2)]


def eval_discart(state):
    ta = state[:64]
    board = chess.Board(FEN(state))
    eval = 0
    if board.is_checkmate():
        if turn(board.epd()):
            eval -= 1000
            return eval
        else:
            eval += 1000
            return eval
        # mat_imbalance
    material_imb = 0
    for s in ta:
        material_imb += dict_v[s]
    eval += material_imb
    # mat_adv
    material_adv = 0
    w_bp, b_bp, w_sb, b_sb, w_rp, b_rp, n_o_s, op_s, pc_ws_a, pc_bs_a, pc_a, wk_saf, bk_saf, n_pa, b_pa, p_pa = \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    for s in range(len(ta)):
        piece = str(board.piece_at(s))
        if piece is None:
            n_o_s += 1
            continue
        elif piece == 'B':
            w_bp += 1
            if (s % 2) == 0:
                w_sb += 1
            else:
                b_sb += 1
            if (1 < chess.square_file(s) < 6) and (1 < chess.square_rank(s) < 6):
                b_pa += 1
        elif piece == 'b':
            b_bp += 1
            if (s % 2) == 0:
                w_sb += 1
            else:
                b_sb += 1
            if (1 < chess.square_file(s) < 6) and (1 < chess.square_rank(s) < 6):
                b_pa -= 1
        elif piece == 'R':
            w_rp += 1
        elif piece == 'r':
            b_rp += 1
        elif piece == 'N':
            if knight_outpost(board, s):
                op_s += 1
            if (1 < chess.square_file(s) < 6) and (1 < chess.square_rank(s) < 6):
                n_pa += 1
        elif piece == 'n':
            if knight_outpost(board, s):
                op_s -= 1
            if (1 < chess.square_file(s) < 6) and (1 < chess.square_rank(s) < 6):
                n_pa -= 1
        elif piece == 'P':
            pc_val = pawnchain(board, s)
            pc_a += pc_val[0]
            if pc_val[1] == 1:
                pc_ws_a += 1
            elif pc_val[1] == 0:
                pc_bs_a += 1
            if (2 < chess.square_file(s) < 5) and (2 < chess.square_rank(s) < 5):
                p_pa += 1
        elif piece == 'p':
            pc_val = pawnchain(board, s)
            pc_a -= pc_val[0]
            if pc_val[1] == 1:
                pc_ws_a -= 1
            elif pc_val[1] == 0:
                pc_bs_a -= 1
            if (2 < chess.square_file(s) < 5) and (2 < chess.square_rank(s) < 5):
                p_pa -= 1
        elif piece == 'K':
            wk_saf = k_safe(board, s)
            if board.is_attacked_by(False, s):
                wk_saf -= 5
        elif piece == 'k':
            bk_saf = k_safe(board, s)
            if board.is_attacked_by(True, s):
                bk_saf -= 5
    bp_a = 0
    if (w_bp == 2) and ((b_bp != 2) and b_bp < 2):
        bp_a += 1
    elif (b_bp == 2) and ((w_bp != 2) and w_bp < 2):
        bp_a -= 1
    elif w_bp > b_bp:
        bp_a += 1
    elif w_bp < b_bp:
        bp_a -= 1
    material_adv += bp_a / (64 - n_o_s)
    rp_a = 0
    if (w_rp == 2) and ((b_rp != 2) and b_rp < 2):
        rp_a += 1
    elif (b_rp == 2) and ((w_rp != 2) and w_rp < 2):
        rp_a -= 1
    elif w_rp > b_rp:
        rp_a += 1
    elif w_rp < b_rp:
        rp_a -= 1
    material_adv += rp_a / (64 - n_o_s)
    eval += material_adv
    # pos_adv TO DO
    position_adv = 0
    o_c_b = 0
    if (w_bp == 1) and (b_bp == 1) and (w_sb == 1) and (b_sb == 1):
        position_adv += 2 * (pc_ws_a + pc_bs_a) / (64 - n_o_s)
    l_m = list(board.legal_moves)
    if turn(FEN(state)):
        position_adv += 2 * (len(l_m) / (64 - n_o_s // 2))
    else:
        position_adv -= 2 * (len(l_m) / (64 - n_o_s // 2))
    position_adv += op_s / (64 - n_o_s)
    position_adv += n_pa / (64 - n_o_s) * 0.05
    position_adv += b_pa / (64 - n_o_s) * 0.05
    position_adv += pc_a / (64 - n_o_s)
    position_adv += wk_saf / (64 - n_o_s)
    position_adv -= bk_saf / (64 - n_o_s)
    position_adv += p_pa
    for s in range(64):
        if (23 < s < 56) and (str(board.piece_at(s)) is not None) and (
                ((s % 3 == 0) and not (s % 2 == 0)) or ((s % 4 == 0) and not (s % 8 == 0))):
            if not turn(FEN(state)):
                position_adv += 0.04
            else:
                position_adv -= 0.04
    eval += position_adv
    # space_adv TO DO val mes l'amenaca si el valor de la peca es menor
    c = 0
    c_power = 0
    for s in range(64):
        att_p = list(board.attackers(True, s))
        for att in att_p:
            att = dict_v_2[str(board.piece_at(att))]
        control = len(att_p)
        c += (control - sum(att_p)) / (64 - n_o_s)
        if control != 0:
            c_power += 1
        if 23 < s < 40:
            c += (control - sum(att_p)) / (64 - n_o_s)
    space_adv = c / c_power
    c = 0
    c_power = 0
    for s in range(64):
        att_p = list(board.attackers(False, s))
        for att in att_p:
            att = dict_v_2[str(board.piece_at(att))]
        control = len(att_p)
        c += (control - sum(att_p)) / (64 - n_o_s)
        if control != 0:
            c_power += -1
        if 23 < s < 40:
            c += (control - sum(att_p)) / (64 - n_o_s)
    space_adv = space_adv + (c / c_power)
    eval += space_adv
    # wk safety
    wk_safety = 0

    # bk safety
    bk_safety = 0
    # Return results
    if eval == 0:
        # print(eval)
        return eval
    if turn(board.epd()):
        eval += 0.009
    else:
        eval += -0.009
    # print(eval)
    return eval - 1.4


# eval_discart(fer_df([chess.Board()])[0])
def selection(boards, discart_n, turn, a_c_boards):
    order_best = [[], []]
    selected = [[], []]
    for b in boards:
        order_best[0].append(b)
        if b in a_c_boards:
            order_best[1].append(a_c_boards[a_c_boards.index(b) + 1])
        else:
            order_best[1].append(eval_discart(b))
            a_c_boards.extend([b, order_best[1][-1]])
    for b in boards:
        if turn == 0:
            val = max(order_best[1])
        else:
            val = min(order_best[1])
        ubi = order_best[1].index(val)
        bo = order_best[0][ubi]
        selected[0].append(bo)
        selected[1].append(val)
        order_best[0].remove(bo)
        order_best[1].remove(val)
    selected[0].reverse()
    selected[1].reverse()
    if 0 < discart_n < len(boards):
        selected[0] = selected[0][:-discart_n]
        selected[1] = selected[1][:-discart_n]
    return selected

    # COMPARAR ESTATS:


def compara(a, b):
    s = 0
    comp = []
    for val in a:
        if b[s] == val:
            comp.append(1)
        else:
            comp.append(0)
        s += 1
    return (sum(comp) / 71) * 100


# Retorna una matriu amb dues a dins, la primera els estats i la segona els valors; estan endrecats de "mes semblant" a
# "menys semblant".
def ordena_similaritat(state, memory):
    comparisons = []
    for f in memory:
        comparisons.append(compara(state, f))
    sort_comp = [[], []]
    for c in range(len(comparisons)):
        val = min(comparisons)
        ubi = comparisons.index(val)
        sort_comp[0].append(memory[ubi])
        sort_comp[1].append(comparisons[ubi])
        comparisons.remove(val)
        memory.remove(memory[ubi])
    sort_comp[0].reverse()
    sort_comp[1].reverse()
    return sort_comp

'''
taulells = fit_width([chess.Board()], [20, 20, 4, 3, 2])
# act_taulells = actions(taulells)
np_features = fer_df(taulells)
'''
# Es pot calcuar la similitud:
'''
t1 = random.choice(np_features)
t2 = random.choice(np_features)
print(compara(t1,t2))
'''
# part_features = np_features[
#                 random.randint(0, len(np_features) // 2):random.randint(len(np_features) // 2, len(np_features))]
# t1 = random.choice(np_features)
if False:
    comparativa = ordena_similaritat(t1, part_features)
    states = (comparativa[0])
    s_values = (comparativa[1])
    print(states[1])
    print(s_values[1])
    print(t1)
if False:
    b = 0
    w = 0
    draw = 0
    # for s in np_features[random.randint(0,len(np_features)//2):random.randint(len(np_features)//2,len(np_features))]:
    boards = (full_tree([chess.Board()], 4))
    boards = fer_df(boards)
    for t in boards:
        eval_prima = eval_discart(t)
        if eval_prima < 0:
            b += 1
        elif eval_prima > 0:
            w += 1
        else:
            draw += 1
    total = w + b + draw
    print("Total:   ", total)
    print("White:   ", w, "  ", str(w * 100 / total) + ' %')
    print("Black:   ", b, "  ", str(b * 100 / total) + ' %')
    print("Draw:    ", draw, "    ", str(draw * 100 / total) + ' %')
# Calcula un arbre "reduit" de mida i fondaria controlables, retorna el millor de cada fondaria
if False:
    taulells = fer_df(full_tree([chess.Board()], 1))
    for t in range(5):
        taulells = selection(taulells, len(taulells), (t % 2), [])
        if (t % 2) == 0:
            val = (taulells[1][0])
        else:
            val = (taulells[1][0])
        print(taulells[1])
        print(chess.Board(FEN(taulells[0][taulells[1].index(val)])))
        print()
        taulells = fer_df(full_tree(taulells[0], 1))
# Calcula una linia de tipus MiniMax a fond 1.
if False:
    taulells = fer_df(full_tree([chess.Board()], 1))
    h_taulells = []
    for t in range(400):
        taulells = selection(taulells, len(taulells) - 1, t % 2, [])
        if (t % 2) == 0:
            val = (taulells[1][0])
        else:
            val = (taulells[1][0])
        print(taulells[1])
        taulell = chess.Board(FEN(taulells[0][taulells[1].index(val)]))
        h_taulells.append(taulell)
        print(taulell)
        print()
        taulells = fer_df(full_tree(taulells[0], 1))
        if h_taulells.count(taulell) >= 3:
            print('Draw by repetition.')
            break
        if taulell.is_checkmate():
            print('Checkmate. ')
            break
# Calcula una linia de tipus MiniMax a fond 2.
if False:
    taulells = fer_df(full_tree([chess.Board()], 1))
    for a in range(3):
        v_taulells = []
        for t in range(len(taulells)):
            ts = fer_df(full_tree([taulells[t]], 1))
            v_taulells.append(selection(ts, len(ts) - 1, t % 2, [])[1])
        print(array(v_taulells))

# MINMAX ATTEMPT 1
'''
def MiniMax(count,depth,taulells,v_taulells,turn,rate,a_c_boards):
    if (count == depth):
        b_value = v_taulells[0]
        #print(count)
        #print(b_value)
        #print(chess.Board(FEN(taulells[0])))
        return b_value
    else:
        if count == 0:
            pass
        elif (count%2 != 0):
            if (turn == 0):
                turn = 1
            elif (turn == 1):
                turn = 0
        else:
            if (turn == 0):
                turn = 1
            elif (turn == 1):
                turn = 0
        count += 1
        if len(taulells) == 1:
            if type(taulells[0]) == list:
                tau = chess.Board(FEN(taulells[0]))
            else:
                tau = taulells[0]
            if tau.is_checkmate():
                if turn == 0:
                    return -1000
                else:
                    return 1000
        ts = fer_df(full_tree(taulells,1))
        rt = len(ts)*(rate-1)//rate
        if rt == 0:
            rt = 1
        sel = selection(ts,rt,turn,a_c_boards)
        ts = sel[0]
        vs = sel[1]
        for s in ts:
            v_taulells.append(MiniMax(count,depth,[s],vs,turn,rate+1,a_c_boards))
        if len(v_taulells) == 0:
            pdb.set_trace()
        if turn == 1:
            b_value = min(v_taulells)
        else:
            b_value = max(v_taulells)
        #print(count)
        return b_value


def sel_minmax(boards, turn):
    a_c_boards= []
    order_best = [[], []]
    selected = [[],[]]
    for b in boards:
        order_best[0].append(b)
        if b in a_c_boards:
            order_best[1].append(a_c_boards[a_c_boards.index(b) + 1])
        else:
            order_best[1].append(MiniMax(0,5,[b],[],turn,10,a_c_boards))
            a_c_boards.extend([b,order_best[1][-1]])
    for b in boards:
        if turn == 0:
            val = min(order_best[1])
        else:
            val = max(order_best[1])
        ubi = order_best[1].index(val)
        bo = order_best[0][ubi]
        selected[0].append(bo)
        selected[1].append(val)
        order_best[0].remove(bo)
        order_best[1].remove(val)
    selected[0].reverse()
    selected[1].reverse()
    return selected
# TO DO :comprova si funciona!
'''


def minmax_beta(board_type, reduction):
    # torn = turn(board_type.epd())
    boards = full_tree([board_type], 1)
    org_bo = slect(boards)
    dat_fs = [org_bo[0][:len(org_bo[0]) // reduction], org_bo[1][:len(org_bo[1]) // reduction]]
    return dat_fs


def slect(boards):
    torn = turn(boards[0].epd())
    boards = fer_df(boards)
    vals = [[], []]
    re_v = [[], []]
    for b in boards:
        vals[0].append(b)
        vals[1].append(eval_discart(b))
    for v in range(len(vals[0])):
        if torn:
            maxim = max(vals[1])
        else:
            maxim = min(vals[1])
        t = vals[0][vals[1].index(maxim)]
        re_v[1].append(maxim)
        re_v[0].append(chess.Board(FEN(t)))
        vals[1].remove(maxim)
        vals[0].remove(t)
    return re_v


def reorder(boards, vals):
    re_boards = [[], []]
    if type(boards) != list:
        torn = turn(boards.epd())
    else:
        #print('TIPO:',type(boards),boards)
        torn = turn(boards[0].epd())
    if type(vals) != list:
        vals = [vals]
    if type(boards) != list:
        boards = [boards]
    for v in range(len(vals)):
        if torn:
            val = max(vals)
        else:
            val = min(vals)
        ind = vals.index(val)
        bo = boards[ind]
        re_boards[0].append(bo)
        re_boards[1].append(val)
        if torn:
            vals.pop(ind)
        else:
            vals.pop(ind)
        boards.pop(ind)
    if re_boards[0] == []:
        pdb.set_trace()
    return re_boards


def stack_minmax(count, depth, b):
    re_boards = [[], []]
    if (count == depth) or (b.is_game_over()):
        val = a_b_pruning(Fen_df(b.epd()))
        print('   ' * count, count)
        return [[b], val]
    elif gen_number(b) in calc_boards:
        val = a_b_pruning(Fen_df(b.epd()))
        print('   ' * count, count)
        return [[b], val]
    else:
        # MOTIU ESTRANY FA QUE re_boards TINGUI VALOR INICIAL (SIGUI EL QUE SIGUI)
        bo_re = minmax_beta(b, 3)
        for bo in range(len(bo_re[0])):
            stack = stack_minmax(count + 1, depth, bo_re[0][bo])
            stack = reorder(stack[0], stack[1])
            for s in range(len(stack[0])):
                st = gen_number(stack[0][s])
                if st not in calc_boards:
                    calc_boards.extend([st,stack[1][s]])
                # if str(st) not in data_file.readline(-1):
                #     data_file.write(str(st))
                #     data_file.write(str(stack[1][s]))
            re_boards[0].extend(stack[0])
            re_boards[1].extend(stack[1])
        #print('Re_boards:',re_boards)
        if re_boards[0] == []:
            #print(b)
            pdb.set_trace()
        r_b = reorder(re_boards[0], re_boards[1]).copy()
        print('   ' * count, count)
        if count == 0:
            pass
        #print('Cosa: ', r_b)
        return r_b


'''
stacked = stack_minmax(0,4,[chess.Board()],[])
for bo in stacked[0]:
    print(bo)
'''


def a_b_pruning(df_type):
    board_type = chess.Board(FEN(df_type))
    brd = gen_number(board_type)
    si = False
    for b in range(len(calc_boards)):
        if brd == calc_boards[b]:
            si = True
    # yea = ((str(brd) in data_file.read()))
    # if (str(brd) in data_file.read()):
        # si = True
    if si:
        # print('Existing value')
        ind = calc_boards.index(brd)
        return calc_boards[ind + 1]
    else:
        # print('New register')
        val = eval_discart(df_type)
        calc_boards.extend([brd, val])
        # data_file.write(str(brd))
        # data_file.write(str(val))
        return val


def gen_number(board_type):
    id_num = 1
    for s in range(64):
        piece = str(board_type.piece_at(s))
        v_p = dict_2[piece]
        '''if v_p == 0:
            id_num = int(str(id_num)+'0')
        else:
            id_num *= v_p * primes[s]'''
        id_num += v_p ** primes[s]
    if turn(board_type.epd()):
        id_num += 1
    return id_num


'''
ts = full_tree([chess.Board()], 1)
vs_l = [[], []]
for t in range(len(ts)):
    std = stack_minmax(0, 3, ts[t], 0.0)
    # std = reorder(std[0],std[1])
    vs_l[0].append(ts[t])
    vs_l[1].append(std[1][0])
vs_l = reorder(vs_l[0], vs_l[1])
vs_l[0].reverse()
vs_l[1].reverse()
for t in range(len(vs_l[0])):
    print(vs_l[0][t])
    print(vs_l[1][t])
'''
'''
print('Stage 2: ')
# '8/8/8/K7/8/RR6/7k/8 w KQkq -'
# stacked = stack_minmax(0, 3, chess.Board('8/8/8/K7/8/RR6/7k/8 w KQkq -'))
before = datetime.datetime.now().time()
stacked = stack_minmax(0, 6, chess.Board())
# pdb.set_trace()
stacked = reorder(stacked[0], stacked[1])
print('Amount of different calculated boards:  ', len(calc_boards)//2)
for bo in range(10):  # range(len(stacked[0])):
    print(stacked[0][-bo])
    print(stacked[1][-bo])
    print()
after = datetime.datetime.now().time()
print(before)
print(after)
'''

# TO DO :acumular minmax_beta a veure si va millor

def main():
    print('Yes.')
    '''
    # print(MiniMax(0,10,fer_df([chess.Board('8/8/8/K7/8/RR6/7k/8 w KQkq -')]),[],0,5,[]))
    
    print(chess.Board('8/8/8/K7/8/RR6/7k/8 w KQkq -'))
    print()
    brds = minmax_beta(chess.Board('8/8/8/K7/8/RR6/7k/8 w KQkq -'))
    for a in range(3):
        bds = []
        for b in brds[0]:
            bds.extend(minmax_beta(b,1))
        brds = []
        for b in bds[0]:
            brds.extend(minmax_beta(b,1))
    print(brds[0][0],brds[1][0])
    
    arb = (full_tree(fer_df([chess.Board()]), 2))
    best = (selection(fer_df(arb), 0, 1, [])[0][0])
    print(chess.Board(FEN(best)))

    seleccio_minmax = [[chess.Board()],[0.0]]
    print(seleccio_minmax[0][0])
    print(seleccio_minmax[1][0])
    for n in range(int(input("Half-moves:  "))):
        seleccio_minmax = minmax_beta(seleccio_minmax[0][0],1)
        print(seleccio_minmax[0][0])
        print(seleccio_minmax[1][0])
    '''

if __name__ == '__main__':
    main()

print_time()
