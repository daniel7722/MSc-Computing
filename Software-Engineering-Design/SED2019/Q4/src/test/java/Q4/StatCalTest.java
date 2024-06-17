package Q4;

import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class StatCalTest {

  private final StatCal statCal = new StatCal();
  @Rule public JUnitRuleMockery context = new JUnitRuleMockery();
  Updatable display = context.mock(Updatable.class);

  @Test
  public void updateDisplayWhenAButtonIsPressed() {
    statCal.addObserver(display);
    context.checking(new Expectations() {{
      exactly(1).of(display).update(2, 2.0);
    }});
    statCal.add(2);
  }

  @Test
  public void updateDisplayCorrectlyWhenMultipleButtonsArePressed() {
    statCal.addObserver(display);
    context.checking(new Expectations() {{
      allowing(display).update(2, 2.0);
      allowing(display).update(4, 3.0);
      exactly(1).of(display).update(6, 4.0);
    }});
    statCal.add(2);
    statCal.add(4);
    statCal.add(6);
  }


}
