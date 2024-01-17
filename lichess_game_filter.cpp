#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

// Function to extract the numeric value from a string
int get_num(const string &s) {
    int num = 0;
    for (char ch : s) {
        if (isdigit(ch)) {
            num = num * 10 + (ch - '0');
        }
    }
    return num;
}

int main(int argc, char* argv[]) {
    // Default file paths and Elo rating thresholds
    string infile = "lichess_db_standard_rated_2023-11.pgn";
    string outfile = "filtered_games.pgn";
    int minelo = 2000;
    int maxelo = 99999;

    // Modify file paths and Elo rating thresholds based on command line arguments
    switch (argc) {
        case 2:
            infile = argv[1];
            break;
        case 3:
            infile = argv[1];
            outfile = argv[2];
            break;
        case 4:
            minelo = stoi(argv[1]);
            infile = argv[2];
            outfile = argv[3];
            break;
        case 5:
            minelo = stoi(argv[1]);
            maxelo = stoi(argv[2]);
            infile = argv[3];
            outfile = argv[4];
            break;
        default:
            cerr << "Usage: ./file_filter [min_elo] [max_elo] [input_file] [output_file]" << endl;
            return 1;
    }

    ifstream inFile(infile);
    ofstream outFile(outfile);

    if (!inFile.is_open() || !outFile.is_open()) {
        cerr << "Error opening files." << endl;
        return 1;
    }

    string curr_game, curr_line, event;
    int elo = maxelo;
    int count = 0;

    while (getline(inFile, curr_line)) {
        if (curr_line.substr(0, 2) == "1.") {
            // Write the game to output file if it meets criteria
            if (elo >= minelo && elo < maxelo && event.find("Blitz") == string::npos && event.find("Bullet") == string::npos) {
                outFile << curr_game << "\n\n";
                count++;
                if (count % 1000 == 0) {
                    cout << "Games processed: " << count << endl;
                }
            }

            // Reset variables for the next game
            curr_game.clear();
            event.clear();
            elo = maxelo;
        } else {
            // Accumulate lines for the current game
            curr_game += curr_line + "\n";

            // Extract game type and Elo ratings
            if (curr_line.find("[Event ") != string::npos) {
                event = curr_line;
            } else if (curr_line.find("Elo") != string::npos) {
                elo = min(get_num(curr_line), elo);
            }
        }
    }

    inFile.close();
    outFile.close();

    cout << "Total games processed: " << count << endl;
    cout << "Output file: " << outfile << endl;
    return 0;
}
