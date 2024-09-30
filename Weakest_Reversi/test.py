from project import  AI
import numpy as np


# def print_chessboard(chessboard):
#     new_chessboard=np.empty([8,8],dtype=str)
#     for x in range(8):
#         for y in range(8):
#             if chessboard[(x, y)] == 1:
#                 new_chessboard[(x, y)] = 'o'
#             elif chessboard[(x, y)] == -1:
#                 new_chessboard[(x, y)] = 'x'
#             else:
#                 new_chessboard[(x, y)] = '.'
#     print(new_chessboard)

def print_chessboard(chessboard):
    STONE_TO_CHAR = {
        0: ' . ',
        -1: ' x ',
        1: ' o ',
    }
    BOARD_SIZE=8
    for row in range(BOARD_SIZE):
        line = []
        for col in range(BOARD_SIZE):
            stone = chessboard[row, col]
            line.append(STONE_TO_CHAR[stone])
        print("%2s %s" % (row, ''.join(line)))

    print("    " + "  ".join(str(e) for e in range(BOARD_SIZE)))

a = np.zeros((8, 8))
a[4, 3] = a[3, 4] = -1
a[3, 3] = a[4, 4] = 1
# b=np.ones((8,8))
ai = AI(8, 1, 5)
# # print(ai.utility(b,1))
# # print(ai.get_action(a,-1))
# ai.go(a)
# actions=ai.get_action(a,-1)
# print(actions)
# for action in actions:
#     new_chessboard=ai.get_result(a,action,-1)
#     print("----------------------------")
#     print_chessboard(new_chessboard)
# # print(a)
# print(ai.count(a,-1))
# print(ai.candidate_list)

def log_to_list():
    dir = "chess_log (1).txt"
    with open(dir, "r") as f:
        data = f.read()
    return np.array(eval(data))

print(log_to_list()[50])

ai.go(log_to_list()[50])
print(ai.candidate_list[1])