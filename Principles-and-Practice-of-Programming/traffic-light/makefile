traffic: main.o trafficLight.o time.o
	g++ -Wall main.o trafficLight.o time.o -o traffic
main.o: main.cpp trafficLight.h time.h
	g++ -Wall -g -c main.cpp
trafficLight.o: trafficLight.cpp trafficLight.h time.h
	g++ -Wall -g -c trafficLight.cpp
time.o: time.cpp trafficLight.h time.h
	g++ -Wall -g -c time.cpp
clean:
	rm *.o
