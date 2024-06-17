package retail;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.List;

public abstract class Order {

  protected final List<Product> items;
  protected final CreditCardDetails creditCardDetails;
  protected final Address billingAddress;
  protected final Address shippingAddress;
  protected final Courier courier;

  protected Order(List<Product> items, CreditCardDetails creditCardDetails, Address billingAddress,
      Address shippingAddress, Courier courier) {
    this.items = items;
    this.creditCardDetails = creditCardDetails;
    this.billingAddress = billingAddress;
    this.shippingAddress = shippingAddress;
    this.courier = courier;
  }

  public void process() {
    process(CreditCardProcessor.getInstance());
  }
  public void process(CardProcessor creditCardProcessor) {
    BigDecimal total = new BigDecimal(0);
    for (Product item : items) {
      total = total.add(item.unitPrice());
    }
    total = getBigDecimal(total);
    creditCardProcessor.charge(round(total), creditCardDetails, billingAddress);
    sendParcel();
  }

  protected BigDecimal round(BigDecimal amount) {
    return amount.setScale(2, RoundingMode.CEILING);
  }

  protected abstract void sendParcel();

  protected abstract BigDecimal getBigDecimal(BigDecimal total);
}
