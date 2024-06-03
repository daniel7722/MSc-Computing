#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <chrono>
#include "sudoku.h"

using namespace std;

/* pre-supplied function to load a Sudoku board from a file */
void load_board(const char* filename, char board[9][9]) {

  cout << "Loading Sudoku board from file '" << filename << "'... ";

  ifstream in(filename);
  if (!in) {
    cout << "Failed!\n";
  }
  assert(in);

  char buffer[512];

  int row = 0;
  in.getline(buffer,512);
  while (in && row < 9) {
    for (int n=0; n<9; n++) {
      assert(buffer[n] == '.' || isdigit(buffer[n]));
      board[row][n] = buffer[n];
    }
    row++;
    in.getline(buffer,512);
  }

  cout << ((row == 9) ? "Success!" : "Failed!") << '\n';
  assert(row == 9);
}

/* internal helper function */
void print_frame(int row) {
  if (!(row % 3)) {
    cout << "  +===========+===========+===========+\n";
  } else {
    cout << "  +---+---+---+---+---+---+---+---+---+\n";
  }
}

/* internal helper function */
void print_row(const char* data, int row) {
  cout << (char) ('A' + row) << " ";
  for (int i=0; i<9; i++) {
    cout << ( (i % 3) ? ':' : '|' ) << " ";
    cout << ( (data[i]=='.') ? ' ' : data[i]) << " ";
  }
  cout << "|\n";
}

/* pre-supplied function to display a Sudoku board */
void display_board(const char board[9][9]) {
  cout << "    ";
  for (int r=0; r<9; r++) {
    cout << (char) ('1'+r) << "   ";
  }
  cout << '\n';
  for (int r=0; r<9; r++) {
    print_frame(r);
    print_row(board[r],r);
  }
  print_frame(9);
}

/* add your functions here */
bool is_complete (char board[9][9]) {
  /*
    This function takes a 9 × 9 array of characters representing a Sudoku board 
    and returns true if all board positions are occupied by digits, and false otherwise.
  */

  // loop over the entire board
  for (int row = 0; row < 9; row++) {
    
    for (int col = 0; col < 9; col++) {

      // check if it's a valid number
      if (board[row][col] > '9' || board[row][col] < '1')
	return false;
      
    }
    
  }

  // the entire board is filled with valid numbers
  return true;
}

bool make_move (const char* position, char digit, char board[9][9]) {
  /* 
     This function attempts to place a digit onto a Sudoku board at a given position.
     If the position is invalid, then return false and the unchanged board
     If the position is valid, then return true and the updated board
  */

  // converting input const char* to integer value for row and column starting from 0 to 8
  int row = position[0] - static_cast<int>('A');
  int col = position[1] - '1';

  // check if the position is out of range
  if ((row > 8) || (row < 0) || (col > 8) || (col < 0))
    return false;

  // check if the position is empty
  if (board[row][col] != '.')
    return false;

  // check if there's identical number in the same row or the same column;
  for (int count = 0; count < 9; count++) {
    
    if (board[count][col] == digit || board[row][count] == digit) 
      return false;
    
  }

  // check if there's identical number in the box
  int row_box = row - row % 3; 
  int col_box = col - col % 3;

  // loop over the box using r and c variables to loop three times
  for (int r = 0; r < 3; row_box++, r++) {
    
    for (int c = 0; c < 3; col_box++, c++) {
      
      if (board[row_box][col_box] == digit)
	return false;
      
    }

    // restore column number to the original one for the next row
    col_box -= 3;
    
  }

  // It is a valid move, so update the board and return true
  board[row][col] = digit;
  return true;
}

bool make_move (int row, int col, char digit, char board[9][9]) {
   /* 
      This function overload takes integer row and integer col parameter instead,
      for a more convenient use in solve_board function
  */
  
  // check if the position is out of range
  if ((row > 8) || (row < 0) || (col > 8) || (col < 0))
    return false;

  // check if the position is empty
  if (board[row][col] != '.')
    return false;

  // check if there's identical number in the same row or the same column;
  for (int count = 0; count < 9; count++) {
    
    if (board[count][col] == digit || board[row][count] == digit) 
      return false;
    
  }

  // check if there's identical number in the box
  int row_box = row - row % 3;
  int col_box = col - col % 3;
 
  for (int r = 0; r < 3; row_box++, r++) {
    
    for (int c = 0; c < 3; col_box++, c++) {
      
      if (board[row_box][col_box] == digit)
	return false;
      
    }

    // restore column number to the original one for the next row
    col_box -= 3;
  }

  // It is a valid move, so update the board and return true
  board[row][col] = digit;
  return true;
}


bool save_board (const char* filename, char board[9][9]) {
  /*
    This function outputs the two-dimensional character array board to a file 
    with name filename. The return value should be true if the file was
    successfully written, and false otherwise.
   */
  
  ofstream out;
  out.open(filename);

  if (out.fail()) {
    
    return false;
    
  }

  // loop over the whole board to output characters
  for (int row = 0; row < 9; row++) {
    
    for (int col = 0; col < 9; col++) {
      
      out.put(board[row][col]);
      
    }

    // end of row character is added once we reach the end of row
    out.put('\n');
    
  }
  
  out.close();
  return true;
}

bool solve_board(char board[9][9]) {
  /*
    This function attempts to solve the Sudoku puzzle in input/output parameter board.
    The return value of the function should be true if a solution is found, in which
    case board should contain the solution found. In the case that a solution does not
    exist the return value should be false and board should contain the original board.
 */

  // initialise recursive count and attempts there is to solve the board
  int recur_count = 0;
  int try_count = 0;
  return solve_board(board, 0, 0, recur_count, try_count);

}
 
bool solve_board(char board[9][9], int row, int col, int &recur, int &try_count){
   /*
     This overload recursive function takes additional row and column parameters that
     specify the start of the solving process
   */
  
  // everytime this function is called, increment recur by 1
  recur++;
  
  // Base case: when all rows have been solved
  if (row == 9) return true;

  // Special case: when reaches the end of row, update row and col number
  if (col == 9) return solve_board(board, row + 1, 0, recur, try_count);

  // only enter loop of making moves if the square we look at is empty 
  if (board[row][col] == '.') {

    // try to put every possible number in that square
    for (char num = '1'; num <= '9'; num++) {
      
      // check if it is a valid move
      if (make_move(row, col, num, board)) {

	// record everytime the program makes a move
	try_count++;

	// if it's a valid move, call solve_board to the next square
	if (solve_board(board, row, col + 1, recur, try_count))
	  return true;

	// if not solved, back track the square to empty so the next number can be tested
	board[row][col] = '.';
	
      }
      
    }
    
  } else { // if the square is not empty, go to the next square
    
    return solve_board(board, row, col + 1, recur, try_count);
    
  }

  // if we try every combination to go forward and it doesn't work, return false to the previous
  // funciton call and fill the next number
  return false;
}


void difficulty (const char* filename, char board[9][9]) {
  /* 
     this function attempts to calculate the difficulty of a unsolved Sudoku board by
     calculate the number of recursive function called, number of attempts  and the
     time taken
  */

  using namespace std::chrono;

  int recur = 0;
  int try_count = 0;

  cout << "Calling load_board():\n";
  load_board(filename, board);

  // start counting the time 
  auto start = high_resolution_clock::now();
  
  if (solve_board(board, 0, 0, recur, try_count)) {

    // finish counting time if it's solved
    auto stop = high_resolution_clock::now();

    // print the results
    cout << filename << " takes " << recur << " recursive function calls to solve the board\n";
    cout << filename << " takes " << try_count << " try to solve the board\n";
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "The time taken to find this solution is " << duration.count() << " microseconds!" << endl;
    
  } else {
    
    cout << "A solution cannot be found.\n";
    
  }
   
}
 
