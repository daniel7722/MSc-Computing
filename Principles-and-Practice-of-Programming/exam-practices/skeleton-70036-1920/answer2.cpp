// CID:
// Degree: MSc Computing Science
// Module: 517 Object Oriented Design and Programming
//
// Add all of your answers to question 2 to this file.

#include<iostream>
#include<string>

// The declarations of the class template 'vector' are below.

template <typename T> class vector {
	public:
		vector(); // constructor that creates an empty vector
		void push_back(const T& item); // adds item to the vector
		vector<T>::constant_iterator cbegin(); // returns constant iterator
		vector<T>::constant_iterator cend(); // returns constant iterator
		unsigned int size(); // returns the number of items
};

// Available helper functions that can be used

/* Returns a std::string that is comprised of the given symbol with the given
 * length. E.g. makeErrorString('#', 3) returns the string: "###" */

std::string makeErrorString(char symbol, unsigned int length);

/* Returns a std::string of the given number with the given decimal places. It
 * also rounds numbers up or down as appropriate. E.g. numberToString(10.9, 0)
 * returns the string: "11" */

std::string numberToString(float number, unsigned int decimal_places);

/* Add your code below this line. */

// Question a

// Question b

int main() {

}
