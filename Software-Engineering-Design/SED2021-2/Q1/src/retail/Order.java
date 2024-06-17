package retail;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.List;

public abstract class Order {

  protected final List<Product> items;
  protected final CreditCardDetails creditCardDetails;
  protected final Address billingAddress;
  protected final Address shippingAddress;
  protected final Courier courier;
  private final Processor processor;
  private BigDecimal total;

  public Order(List<Product> items, CreditCardDetails creditCardDetails, Address billingAddress,
      Address shippingAddress, Courier courier, Processor processor) {
    this.items = Collections.unmodifiableList(items);
    this.creditCardDetails = creditCardDetails;
    this.billingAddress = billingAddress;
    this.shippingAddress = shippingAddress;
    this.courier = courier;
    this.processor = processor;
  }

  public void process() {

    total = new BigDecimal(0);

    for (Product item : items) {
      total = total.add(item.unitPrice());
    }

    total = getTotal(total);

    processor.charge(round(total), creditCardDetails, billingAddress);

    send();
  }

  private BigDecimal round(BigDecimal amount) {
    return amount.setScale(2, RoundingMode.CEILING);
  }

  protected abstract void send();

  protected abstract BigDecimal getTotal(BigDecimal total);
}
