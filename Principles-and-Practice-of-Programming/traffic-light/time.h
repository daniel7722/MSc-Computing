/* time.h - header file for the class Time */

#ifndef TIME_H
#define TIME_H

#include <iostream>

/*********************** Time Class ***************************/

class Time {

private:

  int m_theHour;
  int m_theMins;
  int m_theSecs;

public:
  
  Time();
  /**
   * Constructor that initialises m_theHour, m_theMins, m_theSecs
   */
  Time(int, int, int);

  /**
   * This function adds Time hours, minutes, and second to this->Time. 
   *
   * @param Time is the time we want to add to this->Time
   */
  void add(Time&);

  /** 
   * Overload the << operator to print Time in the format 00:00:00.
   * @param std::ostream&: Output stream
   * @param Time&: reference to Time we want to print
   * @return std::ostream enables continuous usage
   */
  friend std::ostream& operator << (std::ostream&, Time&);

};

#endif // TIME_H
