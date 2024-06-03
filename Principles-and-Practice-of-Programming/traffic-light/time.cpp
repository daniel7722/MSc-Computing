#include "trafficLight.h"
#include "time.h"

#include <iostream>
#include <string>
using namespace std;

Time::Time() : m_theHour(0), m_theMins(0), m_theSecs(0) {}

Time::Time(int hours, int mins, int secs) : m_theHour(hours), m_theMins(mins), m_theSecs(secs) {}

void Time::add(Time& anotherTime) {
  // Sum the seconds and update the member variables
  int secondsSum = m_theSecs + anotherTime.m_theSecs;
  m_theSecs = secondsSum % 60;

  // Sum the minutes and include any carry from seconds
  int minutesSum = m_theMins + anotherTime.m_theMins + (secondsSum / 60);
  m_theMins = minutesSum % 60;

  // Sum the hours and include any carry from minutes
  m_theHour = (m_theHour + anotherTime.m_theHour + (minutesSum / 60)) % 24;
}

ostream& operator<<(ostream& os, Time& t) {
  os << t.m_theHour << ":" << t.m_theMins << ":" << t.m_theSecs;
  return os;
}
