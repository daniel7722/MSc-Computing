package retail;

import java.math.BigDecimal;

public class CreditCardProcessor implements Processor {

  private static final Processor INSTANCE = new CreditCardProcessor();

  private CreditCardProcessor() {}

  public static Processor getInstance() {
    return INSTANCE;
  }

  public void charge(BigDecimal amount, CreditCardDetails account, Address billingAddress) {

    System.out.println("Credit card charged: " + account + " amount: " + amount);

    // further implementation omitted for exam question
  }
}
