package retail;

import java.math.BigDecimal;
import java.util.List;

public class OrderBuilder {

  private List<Product> items = null;
  private CreditCardDetails creditCardDetails = null;
  private Address billingAddress = null;
  private Address shippingAddress = null;
  private Courier courier = null;
  private BigDecimal discount ;
  private boolean giftwrap;
  private Processor processor;

  public OrderBuilder setItems(List<Product> items) {
    this.items = items;
    return this;
  }

  public OrderBuilder setCreditCardDetails(CreditCardDetails creditCardDetails) {
    this.creditCardDetails = creditCardDetails;
    return this;
  }

  public OrderBuilder setBillingAddress(Address billingAddress) {
    this.billingAddress = billingAddress;
    return this;
  }

  public OrderBuilder setShippingAddress(Address shippingAddress) {
    this.shippingAddress = shippingAddress;
    return this;
  }

  public OrderBuilder setCourier(Courier courier) {
    this.courier = courier;
    return this;
  }
  public OrderBuilder setDiscount(BigDecimal discount) {
    this.discount = discount;
    return this;
  }

  public OrderBuilder setGiftwrap(boolean giftwrap) {
    this.giftwrap = giftwrap;
    return this;
  }

  public OrderBuilder setProcessor(Processor processor) {
    this.processor = processor;
    return this;
  }

  public Order build() {
    if (items.size() > 3) {
      return new BulkOrder(items, creditCardDetails, billingAddress, shippingAddress, courier, discount,
          processor);
    }
    return new SmallOrder(items, creditCardDetails, billingAddress, shippingAddress, courier, giftwrap,
        processor);
  }
}