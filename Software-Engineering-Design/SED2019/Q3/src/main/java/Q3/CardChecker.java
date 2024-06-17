package Q3;


import java.util.ArrayList;
import java.util.List;

public class CardChecker {

  private final List<Observer> observerList;

  public CardChecker() {
    observerList = new ArrayList<>();
  }

  public boolean validate(String number) {
    if (number.length() != 12) {
      for (Observer ob: observerList) {
        ob.invalidNumber(number);
      }
      return false;
    }
    return true;
  }

  public int numOfObservers() {
    return observerList.size();
  }

  public void addObserver(Observer observer) {
    observerList.add(observer);
  }

  public void removeObserver(int i) {
    observerList.remove(i);
  }
}
