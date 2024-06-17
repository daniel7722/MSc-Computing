package tennis;

public class TennisScorer {
  TennisView v = new TennisView();
  TennisModel m = new TennisModel();

  private TennisScorer() {
    m.addObserver(v);
    v.playerOneAddController(e -> {
      m.playerOneWinsPoint();
    });
    v.playerTwoAddController(e -> {
      m.playerTwoWinsPoint();
    });
  }

  public static void main (String[] args) {
    new TennisScorer();
  }
}