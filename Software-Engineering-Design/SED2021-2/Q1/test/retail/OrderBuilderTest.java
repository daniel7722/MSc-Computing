package retail;


import java.math.BigDecimal;
import java.util.List;
import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class OrderBuilderTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();

  Processor processor = context.mock(Processor.class);
  Courier courier = context.mock(Courier.class);
  private final Address billing = new Address("here");
  private final List<Product> items = List.of(new Product("Mouse", new BigDecimal("10.00")));

  @Test
  public void smallOrderIsChargedCorrectly() {
    Order small = new OrderBuilder().setCourier(courier).setBillingAddress(billing)
        .setShippingAddress(billing).setProcessor(processor).setGiftwrap(true).setItems(items)
        .setCreditCardDetails(new CreditCardDetails("1234123412341234", 111)).build();
    context.checking(new Expectations() {{
      exactly
    }});
  }

}