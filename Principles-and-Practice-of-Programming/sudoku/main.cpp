#include <iostream>
#include <cstdio>
#include "sudoku.h"

using namespace std;

int main() {

  char board[9][9];

  /* This section illustrates the use of the pre-supplied helper functions. */
  cout << "============= Pre-supplied functions =============\n\n";

  cout << "Calling load_board():\n";
  load_board("easy.dat", board);

  cout << '\n';
	cout << "Displaying Sudoku board with display_board():\n";
  display_board(board);
  cout << "Done!\n\n";

  cout << "=================== Question 1 ===================\n\n";

  load_board("easy.dat", board);
  cout << "Board is ";
  if (!is_complete(board)) {
    cout << "NOT ";
  }
  cout << "complete.\n\n";

  load_board("easy-solution.dat", board);
  cout << "Board is ";
  if (!is_complete(board)) {
    cout << "NOT ";
  }
  cout << "complete.\n\n";

  load_board("medium.dat", board);
  cout << "Board is ";
  if (!is_complete(board)) {
    cout << "NOT ";
  }
  cout << "complete.\n\n";

  load_board("mystery1.dat", board);
  cout << "Board is ";
  if (!is_complete(board)) {
    cout << "NOT ";
  }
  cout << "complete.\n\n";

  load_board("mystery2.dat", board);
  cout << "Board is ";
  if (!is_complete(board)) {
    cout << "NOT ";
  }
  cout << "complete.\n\n";

  load_board("mystery3.dat", board);
  cout << "Board is ";
  if (!is_complete(board)) {
    cout << "NOT ";
  }
  cout << "complete.\n\n";


  cout << "=================== Question 2 ===================\n\n";

  load_board("easy.dat", board);

  // Should be OK 43
  cout << "Putting '1' into I8 is ";
  if (!make_move("I8", '1', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;

  // Edge case 1: row out of range
  cout << "Putting '2' into J8 is ";
  if (!make_move("J8", '2', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;
  
  // Edge case 2: col out of range
  cout << "Putting '1' into I0 is ";
  if (!make_move("I0", '1', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;
  
  // Edge case 3: not empty
  cout << "Putting '1' into B2 is ";
  if (!make_move("B2", '4', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;
  
  // Edge case 4: same number in a row
  cout << "Putting '9' into D5 is ";
  if (!make_move("D5", '9', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;
  
  // Edge case 5: same number in a column
  cout << "Putting '1' into D7 is ";
  if (!make_move("D7", '1', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;

  //Edge case 6: same number in a box 95
  cout << "Putting '4' into G4 is ";
  if (!make_move("G4", '4', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;
  
  cout << "=================== Question 3 ===================\n\n";

  load_board("easy.dat", board);
  if (save_board("easy-copy.dat", board)) {
    cout << "Save board to 'easy-copy.dat' successful.\n";
  } else {
    cout << "Save board failed.\n";
  }
  cout << '\n';

  cout << "=================== Question 4 ===================\n\n";

  load_board("easy.dat", board);
  
  cout << "Putting '1' into I8 is ";
  if (!make_move(8, 7, '1', board)) {
    cout << "NOT ";
  }
  cout << "a valid move. The board is:\n";
  display_board(board);
  cout << endl;
  
  if (solve_board(board)) {
    cout << "The 'easy' board has a solution:\n";
    display_board(board);
  } else {
    cout << "A solution cannot be found.\n";
  }
  cout << '\n';

  load_board("medium.dat", board);
  if (solve_board(board)) {
    cout << "The 'medium' board has a solution:\n";
    display_board(board);
  } else {
    cout << "A solution cannot be found.\n";
  }
  cout << '\n';

  // write more tests

  cout << "=================== Question 5 ===================\n\n";

  // write more tests
  difficulty("easy.dat", board);
  cout << endl;

  difficulty("medium.dat", board);
  cout << endl;

  difficulty("mystery1.dat", board);
  cout << endl;

  difficulty("mystery2.dat", board);
  cout << endl;

  difficulty("mystery3.dat", board);
  cout << endl;
  
  return 0;
}
