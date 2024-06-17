package retail;

import java.math.BigDecimal;

public class CreditCardProcessor implements CardProcessor {

  private static final CardProcessor INSTANCE = new CreditCardProcessor();

  private CreditCardProcessor() {}

  public static CardProcessor getInstance() {
    return INSTANCE;
  }

  @Override
  public void charge(BigDecimal amount, CreditCardDetails account, Address billingAddress) {

    System.out.println("Credit card charged: " + account + " amount: " + amount);

    // further implementation omitted for exam question
  }
}
