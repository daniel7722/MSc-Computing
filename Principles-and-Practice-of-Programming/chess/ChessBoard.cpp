#include <iostream>
#include <cstring>
#include <string>
#include <map>

#include "ChessBoard.h"
#include "piece.h"

using namespace std;

ChessBoard::ChessBoard () {
  inCheck = false;
  validMove = true;

  // initialise map
  pieceMap.insert(make_pair('K', new King('K')));
  pieceMap.insert(make_pair('Q', new Queen('Q')));
  pieceMap.insert(make_pair('R', new Rook('R')));
  pieceMap.insert(make_pair('B', new Bishop('B')));
  pieceMap.insert(make_pair('N', new Knight('N')));
  pieceMap.insert(make_pair('P', new Pawn('P')));
  pieceMap.insert(make_pair('k', new King('k')));
  pieceMap.insert(make_pair('q', new Queen('q')));
  pieceMap.insert(make_pair('r', new Rook('r')));
  pieceMap.insert(make_pair('b', new Bishop('b')));
  pieceMap.insert(make_pair('n', new Knight('n')));
  pieceMap.insert(make_pair('p', new Pawn('p')));
};


ChessBoard::~ChessBoard () {
  for (auto it = pieceMap.begin(); it != pieceMap.end(); it++) {
    Piece* ptr = it->second;
    delete ptr;
  }
  pieceMap.clear();
}

void ChessBoard::loadState (const char* state) {
  int countWS = 0;
  int countSlash = 0;
  int column = 0;
  int canCastleIndex = 0;

  // loop through the information string
  for (;*state != '\0'; state++) {
    // count white spaces
    if (*state == ' ') {
      countWS++; 
    } else if (countWS == 1) { // set the active colour
      activeColour = *state;
    } else if (countWS == 2) { // set the castling availability
      canCastle[canCastleIndex++] = *state;
    } else if (*state == '/') { // encounter a slash indicate next row
      // reset column
      column = 0;
      countSlash++;
    } else { // piece placement information
      if (isdigit(*state)) { // continuous empty square indicated by digits
	int emptyStart = column + (*state - '0');
	for (;column < emptyStart; column++) {
	  // setting emptysquare to null pointer
	  board[countSlash][column] = NULL;
	}
      } else { // piece symbol
	// find the key value pair in the pieceMap
	map<char, Piece*>::iterator it = pieceMap.find(*state);
	Piece *p = it->second;
	// put it on the appropriate square
	board[countSlash][column] = p;
	column++;
      }
    }
  }
  cout << "A new board state is loaded!" << endl;
};

void ChessBoard::print() const {
  cout << "  ---------------------------------" << endl;
  for (int row = 0; row < 8; row++) {
    cout << 8-row << " | ";
    for (int col = 0; col < 8; col++) {
      if (board[row][col]) {
	cout << board[row][col]->getSymbol() << " | ";
      } else {
	cout << "  | ";
      }
    }
    cout << endl;
    cout << "  ---------------------------------" << endl;      
  }
  cout << "    A   B   C   D   E   F   G   H " << endl;
};

bool ChessBoard::canCheck(Piece* board[8][8], char side) {
  // loop over the entire board
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      // a piece of the desired colour on the square
      if (board[row][col] && board[row][col]->getColour() != side) {
	// Check that piece can see the opponent's king
	if (board[row][col] && board[row][col]->checkKing(board, row, col, 0)) 
	  return true;
      }
    }
  }
  // no piece can see the opponent's king
  return false;
};

bool ChessBoard::checkMove(Piece* board[8][8]) {
  // loop over the entire board
  for (int row = 0; row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      // a piece opposite of active colour on the square
      if (board[row][col] && board[row][col]->getColour() != activeColour) {
	// Check if that piece has any valid move
	if (board[row][col]->canMove(board, row, col))
	  return true;
      }
    }
  }
  // no valid move for every piece
  return false;
};

void ChessBoard::submitMove(const char* initial, const char* destination) {
  // check initial square is different from destination
  if (initial == destination) {
    cout << "Same square!" << endl;
    return;
  }
  
  // Convert input into numeric row and column
  int initCol = initial[0] - 65;
  int initRow = 8 - (initial[1] - '0');
  int desCol = destination[0] - 65;
  int desRow = 8 - (destination[1] - '0'); 

  // Check if the initial and destination squares are valid
  if (initRow < 0 ||
      initRow > 7 ||
      initCol < 0 ||
      initCol > 7 ||
      desCol < 0 ||
      desCol > 7 ||
      desRow < 0 ||
      desRow > 7) {
    cout << "Not a valid move!" << endl;
    return;
  }

  // Check if there is a piece at initial square  
  if (!board[initRow][initCol]) {
    cout << "There is no piece at position " << initial << "!" << endl;
    return;
  } else if (board[initRow][initCol]->getColour() != activeColour) { // check it's piece with the right colour moving
    if (activeColour == 'w') {
    cout << "It is not Black's turn to move!" << endl;
    return;
    } else {
      cout <<  "It is not White's turn to move!" << endl;
    }
  }

  // The desired square has a friendly piece
  if (board[desRow][desCol] &&
      board[desRow][desCol]->getColour() == activeColour) {
    cout << "The destination square has a friendly piece!" << endl;
    return;
  }

  // Initialise a piece pointer to null to store if there's piece being captured
  Piece* captured = NULL;
  // Check if the move obeys the piece mechanism, if it does make the move
  if (board[initRow][initCol]->move(board, initRow, initCol, desRow, desCol, captured)) {
    // clear the initial square to NULL;
    board[initRow][initCol] = NULL;
    
    // if king is in check after the move
    if (canCheck(board, activeColour)) {
      // restore the move on the board and leave the function
      board[initRow][initCol] = board[desRow][desCol];
      board[desRow][desCol] = captured;
      return;
    }
    // king is not in check
    inCheck = false;

    // white's turn
    if (activeColour == 'w') {
      // Describe the movement
      cout << "White's " << board[desRow][desCol]->getPieceName() << " moves from " << initial << " to " << destination;
      if (captured)
	cout << " taking Black's " << captured->getPieceName();
      cout << endl;
      
      // see if white's move check black
      inCheck = canCheck(board, 'b');
      // calculate if black has any valid move
      validMove = checkMove(board);
      if (inCheck) {
	cout << "Black is in check";
	if (!validMove) // no valid move and in check -> checkmate
	  cout << "mate";
	cout << endl;
      } else { 
	if (!validMove) // no valid move and no in check -> stalemate
	  cout << "Stalemate" << endl;
      }
      // switch side
      activeColour = 'b';
    } else { // black's turn
      // Describe the movement
      cout << "Black's " << board[desRow][desCol]->getPieceName() << " moves from " << initial << " to " << destination;
      if (captured)
	cout << " taking White's " << captured->getPieceName();
      cout << endl;

      // see if black's move check white
      inCheck = canCheck(board, 'w');
      validMove = checkMove(board);
      if (inCheck) {
	cout << "White is in check";
	if (!validMove) // no valid move and in check -> checkmate
	  cout << "mate";
	cout << endl;
      } else {
	if (!validMove) // no valid move and no in check -> stalemate
	  cout << "Stalemate" << endl;
      }
      // switch side
      activeColour = 'w';
    }
  } else { // the piece doesn't obey the piece mechanism
    if (activeColour == 'w')
      cout << "White";
    else
      cout << "Black";
    cout << "'s " << board[initRow][initCol]->getPieceName() << " cannot move to " << destination << "!"<< endl;
  }
};
