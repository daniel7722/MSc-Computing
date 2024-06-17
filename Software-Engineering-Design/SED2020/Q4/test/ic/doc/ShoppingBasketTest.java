package ic.doc;

import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class ShoppingBasketTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();
  PaymentAdapter payAdapter = context.mock(PaymentAdapter.class);
  private final Item item = new Item("daneil", 50);
  ShoppingBasket shoppingBasket = new ShoppingBasket(payAdapter);

  @Test
  public void checkoutPassTransactionInformationToPaymentProcessor() {
    context.checking(new Expectations(){{
      exactly(1).of(payAdapter).pay("1234123412341234", 0);
    }});
    shoppingBasket.enterCardDetails("1234123412341234");
    shoppingBasket.checkout();
  }

  @Test
  public void addItemIncrementTotalPounds() {
    context.checking(new Expectations() {{
      exactly(1).of(payAdapter).pay("1234123412341234", 50);
    }});
    shoppingBasket.enterCardDetails("1234123412341234");
    shoppingBasket.addItem(item);
    shoppingBasket.checkout();
  }
}
