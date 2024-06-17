// CID:
// Degree: MSc Computing Science
// Module: 517 Object Oriented Design and Programming
//
// Add all of your answers to question 1 to this file.

#include <functional>
void ingestEvent(double value, double latitude, double longitude, int timestamp);
void addObserverToSensorAtPosition(std::function<void()> observer, double latitude,
		double longitude);

int main() {
	// Example for event ingestions
	ingestEvent(1709.88, 988.456, 3470.3, 1);
	ingestEvent(4856.02, 4687.31, 216.378, 2);
	ingestEvent(183.283, 780.31, 1854.87, 3);
	ingestEvent(1046.07, 3325.91, 2326.46, 4);
	ingestEvent(3768.23, 106.577, 978.304, 5);
	ingestEvent(2834.78, 40.3523, 3343.87, 6);
	ingestEvent(3690.92, 578.758, 2952.94, 7);
	ingestEvent(4051.8, 1484.58, 2824.1, 8);

	///////////////////////////////////////////////
	// Q1 b): Your code to add observers goes here:

	return 0;
}

//////////////////////////////////////////
// Your implementation for Q1 a) goes here
