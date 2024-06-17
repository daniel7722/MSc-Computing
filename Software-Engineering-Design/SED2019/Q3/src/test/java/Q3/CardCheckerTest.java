package Q3;

import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThat;
import static org.junit.Assert.assertTrue;

import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class CardCheckerTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();

  Observer observer = context.mock(Observer.class);
  private final CardChecker cardChecker = new CardChecker();

  @Test
  public void canAddAndRemoveObserverToCardChecker() {
    assertThat(cardChecker.numOfObservers(), is(0));
    cardChecker.addObserver(observer);
    assertThat(cardChecker.numOfObservers(), is(1));
    cardChecker.removeObserver(1);
    assertThat(cardChecker.numOfObservers(), is(0));
  }

  @Test
  public void validateIfCardNumberIsTwelveDigitsLong() {
    assertTrue(cardChecker.validate("111222333444"));
    assertFalse(cardChecker.validate("1112223334444"));
    assertFalse(cardChecker.validate("1234"));
  }

  @Test
  public void alertObserverIfInvalidCardNumberIsEntered() {
    context.checking(new Expectations() {{
      exactly(1).of(observer).invalidNumber("1234");
    }});

    cardChecker.validate("1234");
  }

  @Test
  public void canRemoveObserver() {

  }
}
