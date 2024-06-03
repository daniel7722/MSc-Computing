#include <iostream>
#include <cctype>
#include <cmath>
#include <string>

#include "piece.h"

//TODO: castling
//TODO: en passent
//TODO: promotion
//TODO: factor code

using namespace std;

Piece::Piece(char _symbol) : symbol(_symbol) {
  if (_symbol - 'a' < 0) { // uppercase = white
    colour = 'w';
  } else { // lower case = black
    colour = 'b';
  }
}

Piece::~Piece() {}

char Piece::getSymbol() {
  return symbol;
}

char Piece::getColour() {
  return colour;
}

string Piece::getPieceName() {
  return pieceName;
}

bool Piece::canMove(Piece* board[8][8], int initRow, int initCol) {
  // initialise a place holder for captured piece
  Piece *captured = NULL;
  // loop over the entire board
  for (int row  = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      // set captured to NULL when we move to new square
      captured = NULL;
      // Check if the square has a friendly piece, if there is, don't enter move function and jump to the next square
      if (board[row][col] && board[row][col]->getColour() == colour) {
	continue;
	// move the king if the square is valid otherwise go to next square
      } else if (move(board, initRow, initCol, row, col, captured)) { 
	board[initRow][initCol] = NULL;
	// check if this move lead to a check in our own king
	if (!checkKingEntireBoard(board, captured, row, col, initRow, initCol))
	  // if there is no check, this is a valid move
	  return true;
      }
    }
  }
  // no valid move
  return false;
}

bool Piece::checkKingEntireBoard (Piece* board[8][8], Piece* &captured, int desRow, int desCol, int initRow, int initCol) {
  // loop through entire board
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      // Make sure there is a piece and that piece is the oposite of current piece's colour because we want to see if our
      // king will be in check
      if (board[row][col] && board[row][col]->getColour() != colour) {
        // For each piece, we check if it can see our king using checkKing function
	if (board[row][col]->checkKing(board, row, col, 0)) {
	  // king in check, restore the board back to what it was before return
	  board[desRow][desCol] = captured;
	  board[initRow][initCol] = this;
	  return true;
	}
      }
    }
  }
  // king not in check so the move is valid. Hence restore board as well
  board[desRow][desCol] = captured;
  board[initRow][initCol] = this;
  return false;
}

King::King(char name) : Piece(name) {
  pieceName = "King";
}

King::~King() {}

bool King::move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) {
  // Check if the destination is within King's range
  if (abs(desRow - initRow) > 1 || abs(desCol - initCol) > 1) 
    return false;
  
  // set up captured piece in order to restore board
  if (!board[desRow][desCol])
    captured = NULL;
  else
    captured = board[desRow][desCol];

  // move the piece
  board[desRow][desCol] = this;
  return true;
}


bool King::checkKing (Piece* board[8][8], int row, int col, int state)  {
  // loop over each direction
  for (int i = 0; i < 8; i++) {
    // set current square towards a direction
    int currentRow = row + direction[i][0];
    int currentCol = col + direction[i][1];
    // enter checking if current square is within bound
    if (currentRow >= 0 && currentRow < 8 && currentCol >= 0 &&	currentCol < 8) {
      // empty square
      if (!board[currentRow][currentCol]) 
	return false;
      // Here, only check if a square has opponent's king
      if (!board[currentRow][currentCol]->getPieceName().compare("King") &&
	  board[currentRow][currentCol]->getColour() != colour)
	return true;
    }
  }
  // every possible square doesn't have opponent king
  return false;
}

Rook::Rook(char name) : Piece(name) {
  pieceName = "Rook";
}

Rook::~Rook() {};

// This function is recursively checking the squares in between initial and destination 
bool Rook::move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) {
  // Base case: Arrive at the square
  if (initRow == desRow && initCol == desCol) {
    // empty square
    if (!board[desRow][desCol])
      captured = NULL;
    else
      captured = board[desRow][desCol];
    // move the piece
    board[desRow][desCol] = this;
    return true;
  }

  // set up rowDirection and colDirection
  int rowDirection, colDirection;
  if (desRow - initRow == 0)
    rowDirection = 0;
  else
    rowDirection = (desRow - initRow) / abs(desRow - initRow);
  if (desCol - initCol == 0)
    colDirection = 0;
  else
    colDirection = (desCol - initCol) / abs(desCol - initCol);

  // a guard that check if row and col direction is a straigh line
  if (rowDirection != 0 && colDirection != 0) {
    return false;
  }
  
  // There's no piece on initial square or the piece is the current moving piece
  if (!board[initRow][initCol] || board[initRow][initCol] == this)
    // go to one square closer to destination square and call this function recursively
    return this->Rook::move(board,initRow+rowDirection, initCol+colDirection, desRow, desCol, captured);
  // There's a piece but not the current moving piece. It obstructs the moving rook so return false
  else 
    return false;
}

// This function is also recursively checking all possible squares a rook can move to
bool Rook::checkKing (Piece* board[8][8], int row, int col, int state)  {
  // Not original square
  if (state == 1) {
    // King is in range
    if (board[row][col] && board[row][col]->getColour() != colour && !board[row][col]->getPieceName().compare("King"))
      return true;
    // every other case returns false
    return false;
  } else { // original square
    // loop over all directions
    for (int i = 0; i < 4; i++) {
      // set up current square that is being checked
      int currentRow = row + direction[i][0];
      int currentCol = col + direction[i][1];
      // current square within boundw
      while (currentRow >= 0 && currentRow < 8 && currentCol >= 0 && currentCol < 8) {
	// call itself and set state = 1
	if (this->Rook::checkKing(board, currentRow, currentCol, 1))
	  return true;
	// current square has other pieces, rook can't move further so break the while loop and check next direction
	else if (board[currentRow][currentCol]) break;
	// square is empty, so go to next square in the same direction
        currentRow += direction[i][0];
	currentCol += direction[i][1];
      }
    }
    // after checking each direction, there is no king so return false
    return false;
  }
}

Bishop::Bishop(char name) : Piece(name) {
  pieceName = "Bishop";
}

Bishop::~Bishop () {}

// This function is recursively checking the squares in between initial and destination 
bool Bishop::move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) {
  // Base case: Arrive at the square
  if (initRow == desRow && initCol == desCol) {
    // empty square
    if (!board[desRow][desCol])
      captured = NULL;
    else
      captured = board[desRow][desCol];
    // move the piece
    board[desRow][desCol] = this;
    return true;
  }
  
  // set up rowDirection and colDirection
  int rowDirection, colDirection;
  if (desRow - initRow == 0)
    rowDirection = 0;
  else
    rowDirection = (desRow - initRow) / abs(desRow - initRow);
  if (desCol - initCol == 0)
    colDirection = 0;
  else
    colDirection = (desCol - initCol) / abs(desCol - initCol);

  // a guard that check if row and col direction is a diagonal line
  if (abs(rowDirection) != abs(colDirection)) {
    return false;
  }
  
  // There's no piece on initial square or the piece is the current moving piece
  if (!board[initRow][initCol] || board[initRow][initCol] == this)
    // go to one square closer to destination square and call this function recursively
    return this->Bishop::move(board,initRow+rowDirection, initCol+colDirection, desRow, desCol, captured);
  // There's a piece but not the current moving piece. It obstructs the moving rook so return false
  else
    return false;
}

// This function is also recursively checking all possible squares a bishop can move to
bool Bishop::checkKing (Piece* board[8][8], int row, int col, int state) {
  // Not original square
  if (state == 1) {
    // King is in range
    if (board[row][col] && board[row][col]->getColour() != colour && !board[row][col]->getPieceName().compare("King"))
      return true;
    // every other case return false
    return false;
  } else { // original square
    // loop over all directions
    for (int i = 0; i < 4; i++) {
      // set up current square that is being checked
      int currentRow = row + direction[i][0];
      int currentCol = col + direction[i][1];
      // current square is within bound
      while (currentRow >= 0 && currentRow < 8 && currentCol >= 0 && currentCol < 8) {
	// call itself and set state = 1
	if (this->Bishop::checkKing(board, currentRow, currentCol, 1))
	  return true;
	// current square has other pieces, rook can't move further so break the while loop and check next direction
	else if (board[currentRow][currentCol]) break;
	// square is empty, so go to next square in the same direction
        currentRow += direction[i][0];
	currentCol += direction[i][1];
      }
    }
    // after checking each direction, there is no king so return false
    return false;
  }
}

Queen::Queen(char name) : Piece(name) {
  // initialise Rook and Bishop subordinates with the correct colour
  if (colour == 'w') {
    r = new Rook('R');
    b = new Bishop('B');
  } else {
    r = new Rook('r');
    b = new Bishop('b');
  }
  pieceName = "Queen";
}

Queen::~Queen () {
  // delete subordinates
  delete r;
  delete b;
}

bool Queen::move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) {
  // set up row direction and column direction
  int rowDirection, colDirection;
  if (desRow - initRow == 0)
    rowDirection = 0;
  else
    rowDirection = (desRow - initRow) / abs(desRow - initRow);
  if (desCol - initCol == 0)
    colDirection = 0;
  else
    colDirection = (desCol - initCol) / abs(desCol - initCol);

  // Diagonal movement
  if (abs(rowDirection) == abs(colDirection)) {
    // if bishop subordinate can reach move to the destination, queen can 
    if (b->Bishop::move(board,initRow+rowDirection, initCol+colDirection, desRow, desCol, captured)) {
      // move queen to the destination square
      board[desRow][desCol] = this;
      return true;
    } else
      return false;
  } else if (rowDirection != 0 && colDirection != 0) { // not a rook move nor a bishop move
    return false;
  } else { // a rook move
    if (r->Rook::move(board,initRow+rowDirection, initCol+colDirection, desRow, desCol, captured)) {
      // if rook subordinate can move to the destination, move queen there
      board[desRow][desCol] = this;
      return true;
    }
  }
  return false;
}

bool Queen::checkKing (Piece* board[8][8], int row, int col, int state) {
  // return true if either subordinates can check the king
  bool StraightLines = r->Rook::checkKing(board, row, col, state);
  bool Diagonals = b->Bishop::checkKing(board, row, col, state);
  return StraightLines || Diagonals;
}

Knight::Knight(char name) : Piece(name) {
  pieceName = "Knight";
}

Knight::~Knight () {}

bool Knight::move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) {
  // set up row direction and col direction
  int rowDirection = abs(desRow - initRow);
  int colDirection = abs(desCol - initCol);
  // check if the move obeys the knight jump mechanism
  if ((rowDirection == 2 && colDirection == 1) || (rowDirection == 1 && colDirection == 2)) {
    // empty square
    if (!board[desRow][desCol])
      captured = NULL;
    else
      captured = board[desRow][desCol];
    // move the piece
    board[desRow][desCol] = this;
    return true;
  }
  return false;
}

bool Knight::checkKing (Piece* board[8][8], int row, int col, int state) {
  int currentRow, currentCol;
  // loop through all directions
  for (int i = 0; i < 8; i++) {
    // setup the square for checking king
    currentRow = row + direction[i][0];
    currentCol = col + direction[i][1];
    // current square is within bound
    if (currentRow >= 0 && currentRow < 8 && currentCol >= 0 &&	currentCol < 8) {
      // see the opponent king
      if (board[row][col] && board[row][col]->getColour() != colour && !board[row][col]->getPieceName().compare("King"))
	return true;
    }
  }
  // no king spotted for all possible movement
  return false;
}

Pawn::Pawn(char name) : Piece(name) {
  pieceName = "Pawn";
}

Pawn::~Pawn () {}		    

// en passent and promotion haven't added
bool Pawn::move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) {
  // set up row direction for each colour
  int direction, originalRow;
  if (colour == 'w') {
    direction = -1;
    originalRow = 6;
  } else {
    direction = 1;
    originalRow = 1;
  }
  // First case: marching forward
  if (desCol - initCol == 0) {
    // Intends to move 2 squares
    if (abs(initRow - desRow) == 2) {
      // check if it's at its original square
      if (initRow == originalRow) {
	// loop two times to move the pawn one square at a time to check no other pieces in the way
	for (int i = 1; i < 3; i++) {
	  if (board[initRow+(i*direction)][initCol])
	    return false;
	}
	// no piece in the way, so move pawn
	board[desRow][desCol] = this;
	return true;
      } else // pawn not in original square, so can't move 2 squares
	return false;
    } else if (abs(initRow - desRow) == 1) { // Intend to move one square
      // if there is piece obstructing, return false
      if (board[initRow+direction][initCol])
	return false;
      // no piece obstructing, move pawn and return true
      board[desRow][desCol] = this;
      return true;
    }

  // Second case: capturing piece
  } else if (abs(desCol - initCol) == 1) {
    // empty square
    if (!board[desRow][desCol])
      return false; // have to take account for en passent but that's later
    if (initRow + direction == desRow) {
      captured = board[desRow][desCol];
      board[desRow][desCol] = this;
      return true;
    }
  }
  return false;
};

// This function is also recursively checking all possible squares a pawn can move to
bool Pawn::checkKing (Piece* board[8][8], int row, int col, int state) {
  // Not original square
  if (state == 1) {
    // King is in range
    if (board[row][col] && board[row][col]->getColour() != colour && !board[row][col]->getPieceName().compare("King"))
      return true;
    // every other case return false
    return false;
  } else { // original square
    // assign diagonal directions to pawn with specified colour
    if (colour == 'w') {
      direction[0][0] = -1;
      direction[0][1] = -1;
      direction[1][0] = -1;
      direction[1][1] = 1;
    } else {
      direction[0][0] = 1;
      direction[0][1] = -1;
      direction[1][0] = 1;
      direction[1][1] = 1;
    }

    // loop over two directions
    for (int i = 0; i < 2; i++) {
      // set u pcurrenct square
      int currentRow = row + direction[i][0];
      int currentCol = col + direction[i][1];

      // current square within bound
      if (currentRow >= 0 && currentRow < 8 && currentCol >= 0 && currentCol < 8)
	// call itself and determine if there an opponent king on the square
	return this->Pawn::checkKing(board, currentRow, currentCol, 1);
    }
  }
  return false;
}
