package Q3;

public class BoxOffice {
  private final Theatre theatre;
  private final Payments payment;
  private final WaitingList waitList;

  public BoxOffice(Theatre theatre, Payments payment, WaitingList waitList) {
    this.theatre = theatre;
    this.payment = payment;
    this.waitList = waitList;
  }


  public void bookTickets(Show show, int num, Customer cus) {
    if (theatre.checkAvailable(show, num)) {
      payment.pay(show.price()*num, cus);
      return;
    }
    waitList.add(cus, show, num);
  }

  public void returnTickets(Show show, int num) {
      int number = waitList.anyoneWaiting(show, num);
      Customer cus = waitList.whoWaiting(show, num);
      bookTickets(show, number, cus);
  }
}
