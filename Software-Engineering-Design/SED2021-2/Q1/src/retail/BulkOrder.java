package retail;

import java.math.BigDecimal;
import java.util.List;

public class BulkOrder extends Order {
  private final BigDecimal discount;

  public BulkOrder(
      List<Product> items,
      CreditCardDetails creditCardDetails,
      Address billingAddress,
      Address shippingAddress,
      Courier courier,
      BigDecimal discount, Processor processor) {
    super(items, creditCardDetails, billingAddress, shippingAddress, courier, processor);
    this.discount = discount;
  }
  @Override
  protected void send() {
    courier.send(new Parcel(items), shippingAddress);
  }

  @Override
  protected BigDecimal getTotal(BigDecimal total) {
    if (items.size() > 10) {
      total = total.multiply(BigDecimal.valueOf(0.8));
    } else if (items.size() > 5) {
      total = total.multiply(BigDecimal.valueOf(0.9));
    }

    total = total.subtract(discount);
    return total;
  }
}
