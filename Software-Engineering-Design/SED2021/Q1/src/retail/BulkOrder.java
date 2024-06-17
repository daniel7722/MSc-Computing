package retail;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.List;

public class BulkOrder extends Order{
  private final BigDecimal discount;

  public BulkOrder(
      List<Product> items,
      CreditCardDetails creditCardDetails,
      Address billingAddress,
      Address shippingAddress,
      Courier courier,
      BigDecimal discount) {
    super(items, creditCardDetails, billingAddress, shippingAddress, courier);
    this.discount = discount;
  }

  public void process() {
    BigDecimal total = new BigDecimal(0);
    for (Product item : items) {
      total = total.add(item.unitPrice());
    }
    total = getBigDecimal(total);
    CreditCardProcessor.getInstance().charge(round(total), creditCardDetails, billingAddress);
    sendParcel();
  }

  protected void sendParcel() {
    courier.send(new Parcel(items), shippingAddress);
  }

  protected BigDecimal getBigDecimal(BigDecimal total) {
    if (items.size() > 10) {
      total = total.multiply(BigDecimal.valueOf(0.8));
    } else if (items.size() > 5) {
      total = total.multiply(BigDecimal.valueOf(0.9));
    }
    total = total.subtract(discount);
    return total;
  }

  protected BigDecimal round(BigDecimal amount) {
    return amount.setScale(2, RoundingMode.CEILING);
  }
}
