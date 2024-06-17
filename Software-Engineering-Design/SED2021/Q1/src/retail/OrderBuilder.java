package retail;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class OrderBuilder {

  private List<Product> items = new ArrayList<>();
  private CreditCardDetails creditCardDetails;
  private Address billingAddress;
  private Address shippingAddress;
  private Courier courier;
  private BigDecimal discount;

  private boolean giftWrap;

  public OrderBuilder withItems(List<Product> items) {
    this.items = items;
    return this;
  }

  public OrderBuilder withItem(Product item) {
    this.items.add(item);
    return this;
  }
  public OrderBuilder withCreditCardDetails(CreditCardDetails creditCardDetails) {
    this.creditCardDetails = creditCardDetails;
    return this;
  }

  public OrderBuilder withBillingAddress(Address billingAddress) {
    this.billingAddress = billingAddress;
    return this;
  }

  public OrderBuilder withShippingAddress(Address shippingAddress) {
    this.shippingAddress = shippingAddress;
    return this;
  }

  public OrderBuilder withCourier(Courier courier) {
    this.courier = courier;
    return this;
  }

  public OrderBuilder withDiscount(BigDecimal discount) {
    this.discount = discount;
    return this;
  }

  public OrderBuilder withGiftWrap(boolean giftWrap) {
    this.giftWrap = giftWrap;
    return this;
  }

  public Order build() {
    if (shippingAddress == null) {
      shippingAddress = billingAddress;
      System.out.println(shippingAddress);
    }
    if (items.size() > 3) {
      // bulk order
      return new BulkOrder(items, creditCardDetails, billingAddress, shippingAddress, courier, discount);
    } else {
      // small order
      return new SmallOrder(items, creditCardDetails, billingAddress, shippingAddress, courier, giftWrap);

    }
  }
}