package ic.doc;

import com.worldpay.CardNumber;
import com.worldpay.CreditCardTransaction;
import com.worldpay.TransactionProcessor;

public class WorldPayAdapter implements PaymentAdapter {

  @Override
  public void pay(String cardNumber, int totalPounds) {
    CardNumber cardNum = new CardNumber(cardNumber);
    new TransactionProcessor().process(new CreditCardTransaction(cardNum, totalPounds, 0));
  }
}
