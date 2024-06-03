chess: ChessBoard.o piece.o ChessMain.o
	g++ -Wall -g ChessBoard.o piece.o ChessMain.o -o chess

ChessMain.o: ChessMain.cpp ChessBoard.h piece.h
	g++ -Wall -g -c ChessMain.cpp

ChessBoard.o: ChessBoard.cpp ChessBoard.h
	g++ -Wall -g -c ChessBoard.cpp

piece.o: piece.cpp piece.h
	g++ -Wall -g -c piece.cpp

clean:
	rm -f *.o chess
