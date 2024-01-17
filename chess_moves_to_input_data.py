import chess

# Configuration
NUM = 1000
NUM_LIST = [2000, 2200, 1800, 1600, 1400, 1200, 1000]
MOVE_SET_NUM = None
DIR = 'full_datasets'

def switch_move(move: str, wht_turn: bool = True, normal_format: bool = False) -> str:
    """
    Adjusts a move string based on the player's turn.
    """
    if not normal_format:
        move = move[1:5]
    if wht_turn:
        return move
    return move[0] + str(9 - int(move[1])) + move[2] + str(9 - int(move[3]))

def switch_player(board_str):
    """
    Flips the board string to represent the perspective of the opposing player.
    """
    lines = [board_str[i:i + 8] for i in range(0, len(board_str), 8)][::-1]
    return ''.join(lines).swapcase()

def get_board_str(board, white_side: bool, auto_flip: bool = True):
    """
    Generates a string representation of the chess board.
    """
    board_str = str(board).replace('\n', '').replace(' ', '')
    return switch_player(board_str) if (auto_flip and not white_side) else board_str

def write_pos(infile, outfile):
    """
    Reads a file of chess moves, processes them, and writes the board positions and moves to a new file.
    """
    with open(infile, 'r') as f, open(outfile, 'w') as file:
        count = 1
        for line in f:
            moves = line.split()
            board = chess.Board()

            for idx, move in enumerate(moves):
                if MOVE_SET_NUM is not None and idx < MOVE_SET_NUM - 1:
                    continue

                board_str = get_board_str(board, board.turn)
                uci_move = switch_move(move, board.turn)
                file.write(f'{board_str} {uci_move}\n')
                board.push(chess.Move.from_uci(move[1:]))

                if count % 1000 == 0:
                    print(count)
                count += 1

if __name__ == '__main__':
    if NUM_LIST:
        for num in NUM_LIST:
            infile = f"{DIR}/labeled_elo_{num}.txt"
            outfile = f"{DIR}/elo_{num}_pos.txt"
            print(f'Processing: {outfile}')
            write_pos(infile, outfile)
    else:
        infile = f"{DIR}/labeled_elo_{NUM}.txt"
        outfile = f"{DIR}/elo_{NUM}_pos.txt"
        write_pos(infile, outfile)
