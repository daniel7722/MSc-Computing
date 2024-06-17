package Q3;

public interface WaitingList {

  void add(Customer customer, Show show, int num);

  int anyoneWaiting(Show show, int num);

  Customer whoWaiting(Show show, int num);
}
