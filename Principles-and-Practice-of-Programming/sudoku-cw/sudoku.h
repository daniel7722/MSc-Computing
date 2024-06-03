#ifndef SUDOKU_H
#define SUDOKU_H

void load_board(const char* filename, char board[9][9]);

void display_board(const char board[9][9]);

bool is_complete (char board[9][9]);

bool make_move (const char* position, char digit, char board[9][9]);

bool make_move (int row, int col, char digit, char board[9][9]);

bool save_board(const char* filename, char board[9][9]);

bool solve_board (char board[9][9]);

bool solve_board(char board[9][9], int row, int col, int &recur_count, int &try_count);

void difficulty (const char*filename, char board[9][9]);


#endif
