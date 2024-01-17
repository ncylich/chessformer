import torch
from torch.utils.data import Dataset, DataLoader

def square_num(sq: str) -> int:
    """
    Converts chess square notation to a numerical index.
    """
    sq = sq.lower()
    return (ord(sq[0]) - ord('a')) + (8 - int(sq[1])) * 8

def parse_pos_lists(list_file, num_pos=None):
    """
    Parses a file containing chess positions and moves, converting them to numerical representations.
    """
    if isinstance(num_pos, float):
        num_pos = int(num_pos)
    if not isinstance(num_pos, int):
        num_pos = int(1e9)  # Default to a large number if not specified

    with open(list_file, 'r') as file:
        pos = [line for i, line in enumerate(file) if i < num_pos or num_pos < 0]

    boards, new_moves = [], []
    for line in pos:
        if not line:
            continue

        board, new_move = line.strip().split()
        piece_to_index = {'.': 1, 'P': 2, 'N': 3, 'B': 4, 'R': 5, 'Q': 6, 'K': 7,
                          'p': 8, 'n': 9, 'b': 10, 'r': 11, 'q': 12, 'k': 13}
        board = [piece_to_index[p] for p in board]  # Convert pieces to integers

        new_move = new_move[:2], new_move[2:]  # Split move into start and end squares
        new_move = square_num(new_move[0]), square_num(new_move[1])  # Convert squares to indices

        boards.append(board)
        new_moves.append(new_move)

    return boards, new_moves

class ChessDataset(Dataset):
    """
    A custom PyTorch Dataset for chess positions and moves.
    """
    def __init__(self, boards, moves):
        self.boards = [torch.tensor(board, dtype=torch.long) for board in boards]
        self.moves = [torch.tensor(mv, dtype=torch.long) for mv in moves]

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        move = self.moves[idx]
        return board, move

def get_dataloader(pos_file, batch_size=32, num_workers=0, num_pos=None):
    """
    Creates dataloaders for training and testing datasets.
    """
    boards, moves = parse_pos_lists(pos_file, num_pos=num_pos)
    dataset = ChessDataset(boards, moves)

    test_len = min(5000, int(len(dataset) * 0.1))
    dataset, testset = torch.utils.data.random_split(dataset, [len(dataset) - test_len, test_len])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)

    return dataloader, testloader

if __name__ == '__main__':
    # Example usage of the get_dataloader function
    dataloader, testloader = get_dataloader('path_to_your_pgn_file.txt')
    print("Dataloader and Testloader created.")
