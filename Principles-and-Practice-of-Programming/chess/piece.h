#ifndef PIECE_H
#define PIECE_H

#include <string>

// ---------Abstract Class---------
class Piece {
 protected:
  // symbol represents pieces on the board when print the board
  char symbol;

  // Full name that will be used to output the movement
  std::string pieceName;

  // either black or white
  char colour;
 
 public:
  /**
   * Constructor: Initialise the colour attribute and symbol
   * 
   * @param char: directly represent symbol of a piece and if it's uppercase, the piece is white and if it's lower
   * the piece is black. 
   */ 
  Piece (char);

  /**
   * Virtual destructor. ensuring the child object is also deleted responsibly
   */
  virtual ~Piece();

  // return "symbol" 
  char getSymbol();

  // return "colour"
  char getColour();

  // return "pieceName"
  std::string getPieceName();

  /**
   * This function check if the specified movement from initial square to the destination square is valid in each specific
   * kind of piece. For example, Rook can only move in a straight line. When entering the function, it should be noted that
   * the destination does not have a friendly piece.
   * 
   * @param Piece* board is the current status and location of pieces on the board
   * @param int initRow is the row of the initial square in digit form
   * @param int initCol is the column of the initial square in digit form
   * @param int desRow is the row of the destination square in digit form 
   * @param int desCol is the column of the destination square in digit form
   * @param Piece* &captured is a pointer of piece that passed by reference because if the move involve capturing a piece, 
   * it needs to be keep in track in case the move is invalid and is needed to be restored
   * @return bool: True if the move is made, false if it's not a valid move 
   */
  virtual bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured) = 0;

  /**
   * Check if the piece's attacking range include the opponent king
   * 
   * @param  Piece* board is the current status and location of pieces on the board
   * @param int initRow is the row of the initial square in digit form
   * @param int initCol is the column of the initial square in digit form
   * @param int state is either 0 (original square) or 1 (new square). It is designed specifically for Rook and Bishop to
   * do recursion check
   * @return bool: True if the piece can see the opponent's king, false otherwise. 
   */
  virtual bool checkKing(Piece* board[8][8], int row, int col, int state) = 0;

  /** 
   * This function is to check if the opponent has any valid move for all its pieces, without leading to a check on its
   * own king.
   *
   * @param  Piece* board is the current status and location of pieces on the board
   * @param int initRow is the row of the initial square in digit form
   * @param int initCol is the column of the initial square in digit form
   * @return bool: true if there is one or more valid moves, false if there is not valid move
   */
  bool canMove(Piece* board[8][8], int initRow, int initCol);

  /** 
   * Helper function for canMove that loops over the board to use checkKing on the right condition.
   * 
   * @param Piece* board[8][8] is the current status and location of pieces on the board
   * @param Piece* &captured is a pointer of piece that passed by reference because if the move involve capturing a piece,
   * it needs to be keep in track in case the movbe is invalid and is needed to be restored
   * @param int desRow is the row of the square the piece has moved to
   * @param int desCol is the column of the square the piece has moved to
   * @param int initRow is the row of the square the piece has moved from 
   * @param int initCol is the column of the square the piece has moved from 
   * @return bool: true if there is any piece that is checking the king, false when no pieces are checking the king
   */ 
  bool checkKingEntireBoard (Piece* board[8][8], Piece* &captured, int desRow, int desCol, int initRow, int initCol);
};


// -------------Child Classes for Piece---------------
class King : public Piece {
  // King's moving direction
  int direction[8][2] = {
    {1, 1},
    {1, 0},
    {1, -1},
    {0, 1},
    {0, -1},
    {-1, -1},
    {-1, 0},
    {-1, 1}
  };
 public:
  // King's constructor initialising piece name to "King" and symbol to 'k' or 'K'
  King (char);
  ~King ();

  // King moves one square at a time in direction specified
  bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured);
  bool canMove(Piece* board[8][8], int initRow, int initCol);
  bool checkKing(Piece* board[8][8], int row, int col, int state);
};

class Rook : public Piece {
  // Rook's moving direction
  int direction[4][2] = {{1,0}, {-1, 0}, {0, 1}, {0, -1}};
 public:
  // Rook's constructor initialising piece name to "Rook" and symbol to 'r' or 'R'
  Rook (char);
  ~Rook ();

  // Rook moves one or multiple squares in a straight line, either in the same row or in the same column 
  bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured);
  bool canMove(Piece* board[8][8], int initRow, int initCol);
  bool checkKing(Piece* board[8][8], int row, int col, int state);
};

class Bishop : public Piece {
  // Bishop's moving direction
  int direction[4][2] = {{1,1}, {1, -1}, {-1, 1}, {-1, -1}};
 public:
  // Bishop constructor initialising piece name to "Bishop" and symbol to 'b' or 'B'
  Bishop (char);
  ~Bishop ();

  // Bishop moves one or multiple squares diagonally
  bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured);
  bool canMove(Piece* board[8][8], int initRow, int initCol);
  bool checkKing(Piece* board[8][8], int row, int col, int state);
};

class Queen : public Piece {
 private:
  // Queen is essentially the combination of a Rook and a Bishop. Therefore, they are initialised so a Queen
  // piece can use their logic function
  Rook *r;
  Bishop *b;
 public:
  // Queen's constructor initialising the piece name to "Queen" and the symbol to 'q' or 'Q' 
  Queen (char);
  ~Queen ();

  // Queen can move both in a straight line or diagonally to one or multiple squares
  bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured);
  bool canMove(Piece* board[8][8], int initRow, int initCol);
  bool checkKing(Piece* board[8][8], int row, int col, int state);
};

class Knight : public Piece {
  // Knight's moving direction
  int direction[8][2] = {
    {1, 2},
    {1, -2},
    {-1, 2},
    {-1, -2},
    {2, 1},
    {2, -1},
    {-2, 1},
    {-2, -1}
  };
 public:
  // Knight's constructor initialising the piece name to "Knight" and the symbol to 'n' or 'N'
  Knight (char);
  ~Knight ();

  // Knight jump in the direction specified
  bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured);
  bool canMove(Piece* board[8][8], int initRow, int initCol);
  bool checkKing(Piece* board[8][8], int row, int col, int state);
};

class Pawn : public Piece {
  // Pawn's direction is specified upon use
  int direction[2][2];
 public:
  // Pawn's constructor initialising the piece name to "Pawn" and the symbol to 'p' or 'P' 
  Pawn (char);
  ~Pawn ();

  // Pawn's direction can be different in different situations. First, if it's marking forward, it is always on the
  // same row. Second, if it's capturing other pieces, it moves one forward square diagonally
  bool move(Piece* board[8][8], int initRow, int initCol, int desRow, int desCol, Piece* &captured);
  bool canMove(Piece* board[8][8], int initRow, int initCol);
  bool checkKing(Piece* board[8][8], int row, int col, int state);
};


#endif
