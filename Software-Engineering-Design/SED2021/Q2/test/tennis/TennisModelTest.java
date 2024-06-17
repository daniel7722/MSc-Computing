package tennis;

import static org.hamcrest.Matchers.is;
import static org.junit.Assert.assertThat;

import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class TennisModelTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();

  Updatable observer = context.mock(Updatable.class);

  TennisModel model = new TennisModel();

  @Test
  public void playerOneWinIncrementPlayerOnesScore() {
    model.playerOneWinsPoint();
    assertThat(model.score(), is("15 - Love"));
  }

  @Test
  public void playerTwoWinIncrementPlayerOnesScore() {
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    assertThat(model.score(), is("Love - 40"));
  }

  @Test
  public void deuce() {
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    assertThat(model.score(), is("Deuce"));
  }

  @Test
  public void advantagePlayer1() {
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    assertThat(model.score(), is("Advantage Player 1"));
  }

  @Test
  public void game() {
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerTwoWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    model.playerOneWinsPoint();
    assertThat(model.score(), is("Game Player 1"));
  }


}