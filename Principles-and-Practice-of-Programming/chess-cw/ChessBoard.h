#ifndef CHESS_BOARD_H
#define CHESS_BOARD_H

#include <map>
#include "piece.h"

// set the boundry of a ChessBoard
const int BOARD_EDGE = 8;

class ChessBoard {
 private:
  // 2D array that record the board status by piece pointer and NULL
  Piece* board[BOARD_EDGE][BOARD_EDGE];

  // active colour can either be 'w' as white and 'b' as black, they are alternating
  char activeColour;

  // 4 stands for white's ability to castle King side 'K', and Queen side 'Q'. And black's ability to castle king side 'k'
  // and queen side 'q'.
  char canCastle[4];

  // Show if the opposite side is in check after each move
  bool inCheck;

  // true if there is more than one valid moves in a side of the game
  bool validMove;

  // a map of piece with it's symbol
  std::map<char, Piece*> pieceMap;
  
 public:
  /** 
   * Constructor: initialise inCheck to false and validMove to true
   */ 
  ChessBoard ();

  /** 
   * Destructor: make sure to deallocate pieceMap
   */ 
  ~ChessBoard ();

  /**
   * This would set up the chess game with pieces on the board and set the availability of castling and active colour
   *
   * @param const char* is a long string with piece placement separated by 7 slashes indicating each row. And a white
   * space followed by the active colour information. Then, a whitespace followed by castling availability
   */
  void loadState (const char*);

  /** 
   * print out the piece pointer board in a readable way
   */
  void print () const;

  /**
   * check if the opposing colour have pieces that can attack king in colour "side"
   * 
   * @param Piece* board is the current piece pointer status and location
   * @param char side is the colour which its king status is being examined for being checked or not
   * @return bool: true if the king in colour "side" is in check, false if it is not. 
   */
  bool canCheck(Piece* board[BOARD_EDGE][BOARD_EDGE], char side);

  /**
   * check if the opponent of active colour side has any valid move
   *
   * @param Piece* board is the current pieces pointer status and location
   * @return bool: true if there is more than one valid move; false if there is no valid move at all
   */
  bool checkMove(Piece* board[BOARD_EDGE][BOARD_EDGE]);

  /**
   * Move a piece from initial square to destination square
   *
   * @param const char* initial is the initial square passed in using standard chess board notation like "E4"
   * @param const char* destination is the destination square the player wish to move the piece to. It is passed in 
   * using the same notation
   */
  void submitMove(const char* initial, const char* destination);

};

#endif
