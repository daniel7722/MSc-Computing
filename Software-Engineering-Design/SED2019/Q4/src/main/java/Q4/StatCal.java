package Q4;

import java.util.ArrayList;
import java.util.List;

public class StatCal {

  private final List<Integer> numbers = new ArrayList<>();
  private final List<Updatable> observers = new ArrayList<>();
  private int max;
  private double mean;

  public void add(int n) {
    numbers.add(n);
    max = Math.max(max, n);
    mean = numbers.stream().mapToInt(val -> val).average().orElse(0.0);
    notifyObservers();
  }

  private void notifyObservers() {
    for (Updatable ob: observers) {
      ob.update(max, mean);
    }
  }

  public void addObserver(Updatable observer) {
    observers.add(observer);
  }
}
