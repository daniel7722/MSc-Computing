package retail;

import java.math.BigDecimal;

public interface Processor {

  void charge(BigDecimal round, CreditCardDetails creditCardDetails, Address billingAddress);
}
