#ifndef TRAFFICLIGHT_H
#define TRAFFICLIGHT_H

#include "time.h"
#include <string>

class TrafficLight {

public:
  // Constructor

  /**
   * Constructor for TrafficLight. 
   * Initialise 'light' to "red"
   * 
   * @param Time initialises 'delay'
   * @param char* initialises 'name' 
   */
  TrafficLight(Time,char*);

  /**
   * Constructor for TrafficLight with collaborating light specified 
   * Set 'light' to "red"
   * 
   * @param Time initialises 'delay'
   * @param char* initialises 'name'
   * @param TrafficLight& initialises '*collab'
   */
  TrafficLight(Time,char*,TrafficLight&);

  // Member Funcitons
  /**
   * Send signal to TrafficLight object and trigger the requests to change light colour.
   * At the end of the function, traffic light will coordinate itself and the car would be able
   * to cross the road.
   */
  void carWantsToCross();

  // Static Function
  /**
   * STATIC function that sets the static member 'globol' so every TrafficLight instances share
   * the same global time
   * 
   * @param Time& sets the static member 'global'
   */
  static void setTheTime(Time&);
  
  /**
   * Overload the << operator to print the name of the TrafficLight.
   * 
   * @param std::ostream& Output stream. 
   * @param TrafficLight* Pointer to TrafficLight object.
   * @return std::ostream& enables continuous usage
   */
  friend std::ostream& operator << (std::ostream&, TrafficLight*);
  
private:

  //Member variables

  // The colour of light
  std::string light;

  // every light has a delay time for it to change colour
  Time delay;

  // The name of light that indicates the direction it's controlling
  std::string name;
  
  // collaborating light that controls different direction
  TrafficLight *collab;

  // Static member to keep track of the global time for all instances
  static Time global;

  // Private member functions
  /**
   * First wait for the delay time and change the attribute light to specified colour
   * 
   * @param TrafficLight* the TrafficLight to which we want to change light.
   * @param string is the colour the TrafficLight is changing to. 
   */ 
  void turn_light(TrafficLight*, std::string);

  /**
   * Check every condition and decides which traffic light to change and which colour to change to
   * @param TrafficLight* is the TrafficLight to which the car, or other TrafficLight send signal.
   * @param string is the colour the signal sender wants the receiver to change.
   */ 
  void request_change(TrafficLight*, std::string);

};

#endif // TRAFFICLIGHT_H

