package Q3;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class BoxOfficeTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();
  public Theatre theatre = context.mock(Theatre.class);
  public Payments payments = context.mock(Payments.class);
  public WaitingList waitList = context.mock(WaitingList.class);

  static final Show LION_KING =
      new Show("The Lion King", 44.00);

  static final Show HAMILTON =
      new Show("Hamilton", 80.00);

  static final Customer SALLY = new Customer("Sally Davies");
  static final Customer TOM = new Customer("Thomas Williams");
  private final BoxOffice boxoffice = new BoxOffice(theatre, payments, waitList);

  // write your tests here
  @Test
  public void clientCanBookTicketIfAvailable() {
    context.checking(new Expectations() {{
      exactly(1).of(theatre).checkAvailable(LION_KING, 4);will(returnValue(true));
      exactly(1).of(payments).pay(LION_KING.price()*4, SALLY);
    }});
    boxoffice.bookTickets(LION_KING, 4, SALLY);
  }

  @Test
  public void addCustomerToWaitListIfTicketNotAvailable() {
    context.checking(new Expectations() {{
      exactly(1).of(theatre).checkAvailable(HAMILTON, 2);will(returnValue(false));
      exactly(1).of(waitList).add(TOM, HAMILTON, 2);
    }});
    boxoffice.bookTickets(HAMILTON, 2, TOM);
  }

  @Test
  public void ifTicketsReturnedWaitListIsCalled() {
    context.checking(new Expectations() {{
      exactly(1).of(waitList).anyoneWaiting(HAMILTON, 4);will(returnValue(2));
      exactly(1).of(waitList).whoWaiting(HAMILTON, 4);will(returnValue(TOM));
      allowing(theatre).checkAvailable(HAMILTON, 2);will(returnValue(true));
      allowing(payments).pay(HAMILTON.price() * 2, TOM);
    }});
    boxoffice.returnTickets(HAMILTON, 4);

  }

}
