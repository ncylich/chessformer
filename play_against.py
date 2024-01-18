import chess
import torch
from inference_test import preprocess, postprocess_valid
from copy import deepcopy

# Model Configuration
MODEL = "2000_elo_pos_engine_best_test_whole.pth"

# Device setup for model
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = torch.load(f'models/{MODEL}').to(device)

# Initialize the chess board and move history
board = chess.Board()
made_moves = []

# Function to handle player's move input
def get_player_move(board):
    while True:
        move_input = input("Your move (in SAN or UCI format): ")
        try:
            move = board.parse_san(move_input)
        except (chess.InvalidMoveError, chess.IllegalMoveError):
            try:
                move = chess.Move.from_uci(move_input.lower())
            except (chess.InvalidMoveError, chess.IllegalMoveError):
                print("Invalid move. Please try again.")
                continue

        if move in board.legal_moves:
            board.push(move)
            made_moves.append(move.uci())
            break

    return board

# Game Loop
while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
    print(board)
    print("\nMove history:", made_moves)

    # Determining who the AI is playing as
    ai_player = input("Is the AI playing as white (w) or black (b)? ").lower()
    if ai_player in ['b', 'w']:
        ai_player = ai_player == 'b'
        break

    print("Invalid choice. Please enter 'w' for white or 'b' for black.")

# Play as human if AI is set to play as black
if ai_player:
    board = get_player_move(board)

count = 0
while not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
    # AI's turn
    input_tensors = preprocess(board)
    count += 1

    def predict_move(rep_mv=""):
        with torch.no_grad():
            output = model(*input_tensors)
        uci_mv = postprocess_valid(output, board, rep_mv=rep_mv)
        return uci_mv

    uci_move = predict_move()

    # avoiding 3-move repetition
    temp_board = deepcopy(board)
    temp_board.push(chess.Move.from_uci(uci_move))
    if temp_board.can_claim_threefold_repition():
        uci_move = predict_move(rep_mv=uci_move)

    # Prioritize checkmate move if available
    for move in board.legal_moves:
        temp_board = deepcopy(board)
        temp_board.push(move)
        if temp_board.is_checkmate():
            uci_move = move.uci()
            break

    # Execute AI's move
    move = chess.Move.from_uci(uci_move)
    board.push(move)
    made_moves.append(move.uci())

    # Display board and move history
    print(board)
    print(f"Predicted move {count}: {move}")
    print("\nMove history:", made_moves)

    if not (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw()):
        board = get_player_move(board)
