#include "trafficLight.h"
#include "time.h"

#include <iostream>
#include <string>
#include <iomanip>

using namespace std;

// TrafficLight class implementation

TrafficLight::TrafficLight(Time delayT, char* nameT) : delay(delayT), name(nameT) {
  // Every instance of traffic light starts with colour red.
  light = "red";
  collab = NULL;
}

TrafficLight::TrafficLight(Time delayT, char* nameT, TrafficLight& collabLight)
  : delay(delayT), name(nameT), collab(&collabLight) {
  light = "red";
  collabLight.collab = this;
}

void TrafficLight::carWantsToCross () {
  cout << endl;
  cout << "***  at " << global << " a car wants to cross light " << this << ", with colour: ";
  cout << light << endl;
  if (light.compare("red") == 0) {

    // if this light is red and collab light is green, request collab light to red
    if (collab && collab->light.compare("green") == 0)
      request_change(collab, "red");

    // if both lights are red, turn this light to green
    else if (collab && collab->light.compare("red") == 0) {
      turn_light(this, "yellow");
      turn_light(this, "green");
      
    } else { // Error in light colour or collab is NULL
      cerr << "Error: Unexpected state for collaborating traffic light!!" << endl;
    }
    
    // if this light is green, then car passes 
  } else if (light == "green") {
    return;

  } else { // Error in light colour or collab is NULL
    cerr << "Error: Unexpected state for this traffic light!!" << endl;
  }
}

ostream& operator << (ostream& os, TrafficLight* TrafLight) {
  os << TrafLight->name;
  return os;
}

void TrafficLight::turn_light(TrafficLight *TrafLight, string colour) {

  // Increment the global time by delyed time
  global.add(TrafLight->delay);

  // Change colour
  TrafLight->light = colour;

  cout << setw(8) << "at " << global << " " << TrafLight << " changes colour to ";
  cout << TrafLight->light << endl;
}
  

void TrafficLight::request_change(TrafficLight *TrafLight, string colour) {

  // check if the inpur light is valid
  if (!TrafLight) {
    cerr << "Error: Attempting to change an invalid traffic light." << endl;
    return;
  }

  
  if (colour.compare("red") == 0) {

    // Condition where we want to change this light from green to red
    if (TrafLight->light.compare("green") == 0) {

      // First turn yellw
      turn_light(TrafLight, "yellow");

      // Then request the collab light to turn to green
      if (TrafLight->collab)
        request_change(TrafLight->collab, "green");
      
      else { // collab light is NULL
        cerr << "Error: Unexpected state for collaborating traffic light." << endl;
      }
      
    } else if (TrafLight->light.compare("yellow") == 0) { // Change from yellow to red

      // First turn red
      turn_light(TrafLight, "red");

      // Then request the collab light to turn green
      if (TrafLight->collab)
        request_change(TrafLight->collab, "green");
      
      else { // collab light is NULL
        cerr << "Error: Unexpected state for collaborating traffic light." << endl;
      }
      
    } else { // collab light is not green nor yellow
      cerr << "Error: Unexpected state for traffic light." << endl;
    }
    
  } else if (colour.compare("green") == 0) {
    
    // Condition to change red to green
    if (TrafLight->light.compare("red") == 0) {

      // First turn yellow 
      turn_light(TrafLight, "yellow");

      // Then request the collab light to turn red
      if (TrafLight->collab)
        request_change(TrafLight->collab, "red");
      
      else { // collab light is NULL
        cerr << "Error: Unexpected state for collaborating traffic light." << endl;
      }
      
    } else if (TrafLight->light.compare("yellow") == 0) { // change yellow to green

      // Turn green
      turn_light(TrafLight, "green");
       
    } else { // collab light is not red nor yellow
      cerr << "Error: Unexpected state for traffic light." << endl;
    }
  } else { // this light is not in valid colour
    cerr << "Error: Invalid colour specified." << endl;
  }
}

// Initilisation of global time
Time TrafficLight::global(0, 0, 0);

void TrafficLight::setTheTime(Time& t) {
  global = t;
}
