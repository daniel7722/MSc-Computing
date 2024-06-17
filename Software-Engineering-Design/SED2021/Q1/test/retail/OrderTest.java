package retail;

import static org.hamcrest.JMock1Matchers.equalTo;

import java.math.BigDecimal;
import java.util.List;
import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class OrderTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();
  CardProcessor cardProcessor = context.mock(CardProcessor.class);
  Courier courier = context.mock(Courier.class);

  @Test
  public void smallOrderIsChargedCorrectly() {
    CreditCardDetails creditCardDetails = new CreditCardDetails("1234123412341234", 111);
    Address address = new Address("Imperial College");
    Product mouse = new Product("mouse", new BigDecimal("50.00"));
    Product keyboard = new Product("keyboard", new BigDecimal("100.00"));
    List<Product> items = List.of(mouse, keyboard);
    context.checking(new Expectations() {{
      exactly(1).of(courier).deliveryCharge();
      will(returnValue(new BigDecimal(0)));
      exactly(1).of(cardProcessor).charge(new BigDecimal("153.00"), creditCardDetails, address);
      exactly(1).of(courier).send(with(any(GiftBox.class)), with(equal((address))));
    }});

    Order smallOrder = new OrderBuilder().withCourier(courier)
        .withItem(mouse).withItem(keyboard)
        .withCreditCardDetails(creditCardDetails)
        .withBillingAddress(address).withGiftWrap(true).build();
    smallOrder.process(cardProcessor);
  }

  @Test
  public void bulkOrderIsChargedCorrectly() {
    CreditCardDetails creditCardDetails = new CreditCardDetails("1234123412341234", 111);
    Address address = new Address("Imperial College");
    Product mouse = new Product("mouse", new BigDecimal("50.00"));
    Product keyboard = new Product("keyboard", new BigDecimal("100.00"));
    List<Product> items = List.of(mouse, keyboard, mouse, mouse, keyboard, keyboard);
    context.checking(new Expectations() {{
      exactly(1).of(cardProcessor).charge(new BigDecimal("402.00"), creditCardDetails, address);
      exactly(1).of(courier).send(with(any(Parcel.class)), with(equal((address))));
    }});

    Order smallOrder = new OrderBuilder().withCourier(courier)
        .withItems(items)
        .withCreditCardDetails(creditCardDetails)
        .withBillingAddress(address).withDiscount(new BigDecimal("3.00")).build();
    smallOrder.process(cardProcessor);
  }
}
