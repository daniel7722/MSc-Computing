package tennis;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class TennisScorer {

  Model m = new Model();
  View v = new View();

  public static void main(String[] args) {
    new TennisScorer();
  }

  private TennisScorer() {
    m.addObserver(v);
    v.playerOneScoresAddListener(e -> m.playerOneWinsPoint());
    v.playerTwoScoresAddListener(e -> m.playerTwoWinsPoint());
  }
}