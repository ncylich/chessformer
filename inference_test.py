import torch
import chess
from chess_loader import ChessDataset
from chessformer import ChessTransformer
from chess_moves_to_input_data import get_board_str, switch_player, switch_move
from torch.utils.data import DataLoader
from copy import deepcopy
import time

# Configuration
MODEL = "2000_elo_pos_engine_best_test_whole.pth"

# Model and device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(f'models/{MODEL}').to(device)

# Preprocessing function
def preprocess(board):
    """
    Converts a chess board state to a tensor representation.
    """
    board_str = get_board_str(board, white_side=board.turn)
    piece_to_index = {'.': 1, 'P': 2, 'N': 3, 'B': 4, 'R': 5, 'Q': 6, 'K': 7,
                      'p': 8, 'n': 9, 'b': 10, 'r': 11, 'q': 12, 'k': 13}
    board_pieces = [piece_to_index[p] for p in board_str]
    return torch.tensor([board_pieces], dtype=torch.long).to(device)

# Helper functions for postprocessing
def sq_to_str(sq):
    """
    Converts a square index to algebraic notation.
    """
    return chr(ord('a') + sq % 8) + str(8 - sq // 8)

def postprocess_valid(output, board: chess.Board):
    """
    Converts model output to a valid chess move.
    """
    start = time.time()
    single_output = output[0].tolist()
    all_moves = []
    for i, st_sqr in enumerate(single_output):
        for j, end_sq in enumerate(single_output):
            if i != j:
                all_moves.append(((i, st_sqr[0]), (j, end_sq[1])))

    all_moves.sort(key=lambda x: x[0][1] + x[1][1], reverse=True)
    legal_moves = [str(move) for move in board.legal_moves]

    for mv in all_moves:
        mv_str = sq_to_str(mv[0][0]) + sq_to_str(mv[1][0])
        if not board.turn:
            mv_str = switch_move(mv_str, wht_turn=board.turn, normal_format=True)
        if mv_str in legal_moves:
            print(f'Move: {mv_str} = {chess.Board.san(board, chess.Move.from_uci(mv_str))}')
            print('Completed in:', str(time.time() - start))
            return mv_str
    print('Completed in:', str(time.time() - start))
    return None

# Main execution
if __name__ == '__main__':
    # Sample chess game data
    input_data = ["Pe2e4", "pe7e5", "Ng1f3"]
    board = chess.Board()

    for move in input_data:
        mv = move if len(move) == 4 else move[1:]
        board.push(chess.Move.from_uci(mv))

    count = 0
    while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
        input_tensors = preprocess(board)
        count += 1
        with torch.no_grad():
            output = model(*input_tensors)
        uci_move = postprocess_valid(output, board)
        board.push(chess.Move.from_uci(uci_move))
        print(f'Predicted {count}\n', board)
