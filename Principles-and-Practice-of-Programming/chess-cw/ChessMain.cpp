
#include "ChessBoard.h"
#include "piece.h"

#include <iostream>
using namespace std;

int main() {

	cout << "========================\n";
	cout << "Testing the Chess Engine\n";
	cout << "========================\n\n";

	ChessBoard cb;
	cb.loadState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq");
	// cb.print();
	cout << '\n';

	cb.submitMove("D7", "D6");
	// cb.print();
	cout << '\n';
	
	cb.submitMove("D4", "H6");
	// cb.print();
	cout << '\n';
	
	cb.submitMove("D2", "D4");
	// cb.print();
	cout << '\n';
	
	cb.submitMove("F8", "B4");
	// cb.print();
	cout << '\n';

	cout << "=========================\n";
	cout << "Alekhine vs. Vasic (1931)\n";
	cout << "=========================\n\n";

	cb.loadState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq");
	cout << '\n';

	cb.submitMove("E2", "E4");
	// cb.print();
	cb.submitMove("E7", "E6");
	// cb.print();
	cout << '\n';

	cb.submitMove("D2", "D4");
	// cb.print();
	cb.submitMove("D7", "D5");
	//cb.print();
	cout << '\n';

	cb.submitMove("B1", "C3");
	// cb.print();
	cb.submitMove("F8", "B4");
	// cb.print();
	cout << '\n';

	cb.submitMove("F1", "D3");
	// cb.print();
	cb.submitMove("B4", "C3");
	// cb.print();
	cout << '\n';

	cb.submitMove("B2", "C3");
	// cb.print();
	cb.submitMove("H7", "H6");
	// cb.print();
	cout << '\n';

	cb.submitMove("C1", "A3");
	// cb.print();
	cb.submitMove("B8", "D7");
	// cb.print();
	cout << '\n';

	cb.submitMove("D1", "E2");
	// cb.print();
	cb.submitMove("D5", "E4");
	// cb.print();
	cout << '\n';

	cb.submitMove("D3", "E4");
	// cb.print();
	cb.submitMove("G8", "F6");
	// cb.print();
	cout << '\n';

	cb.submitMove("E4", "D3");
	// cb.print();
	cb.submitMove("B7", "B6");
	// cb.print();
	cout << '\n';

	cb.submitMove("E2", "E6");
	// cb.print();
	cb.submitMove("F7", "E6");
	// cb.print();
	cout << '\n';

	cb.submitMove("D3", "G6");
	// cb.print();
	cout << '\n';
	return 0;
}
