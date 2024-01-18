# ChessFormer

ChessFormer is a chess transformer engine developed by [@ncylich](https://github.com/ncylich) that leverages the power of transformer models trained on lichess datasets. Designed to simulate a player with an approximate Elo rating of ~1800, ChessFormer provides a sophisticated approach to chess move prediction and game analysis.

## Project Overview

ChessFormer uses extensive datasets from [lichess](https://lichess.org/), a popular online chess platform, to train the transformer model. The project includes several scripts that handle different aspects of chess data processing and model interaction, providing a comprehensive toolkit for chess enthusiasts and researchers.

## Repository Structure

- `transformer.py`: Implements the chess transformer model.
- `chess_loader.py`: Module for loading and processing chess game datasets.
- `uci_to_pos.py`: Script for converting Universal Chess Interface (UCI) notation to board positions.
- `inference.py`: Contains functions for model inference, including preprocessing and postprocessing.
- `file_filter.cpp`: C++ script to filter chess games from PGN files based on Elo ratings.
- `write_positions.py`: Python script to extract and record specific chess positions from games.
- `models.zip`: The best model weights I could produce with my limited m1 pro setup.

## Setup and Requirements

Before running the scripts, ensure you have Python 3.x and a C++ compiler installed. Follow these steps:

1. Clone the ChessFormer repository:

    ```bash
    git clone https://github.com/ncylich/chessformer
    cd chessformer
    ```

2. Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Compile the C++ script for filtering PGN files:

    ```bash
    g++ -o file_filter file_filter.cpp
    ```

## Usage

- **Chess Transformer Model**: The `chessformer.py` script is the core of ChessFormer, encapsulating the transformer model for move prediction.

- **Data Loading and Processing**: Use `chess_loader.py` for loading and processing chess game data.

- **Notation Conversion**: Convert UCI notations to board positions using `chess_moves_to_input_data .py`.

- **Running the Engine**: Perform model inference with `inference_test.py`, which includes both preprocessing of chess positions and postprocessing of the model's output.
- **Playing Against Engine**: Run `play_against.py`. Must unzip `models.zip` first though.

- **Filtering PGN Files**: Utilize the compiled `file_filter` program to filter games from PGN files based on specific Elo ratings.

Refer to each script's documentation for more detailed instructions.

## Contributing

Feel free to contribute to ChessFormer. You can contribute by:

1. Forking the repository.
2. Creating a new branch for your feature or bug fix.
3. Committing your changes.
4. Pushing your branch and submitting a pull request.

## License

This project is licensed under the [MIT License](https://github.com/ncylich/chessformer/blob/main/LICENSE).

---