package tennis;

import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.WindowConstants;

public class View implements Updatable {

  private final JFrame window = new JFrame("Tennis");
  private final JButton playerOneScores = new JButton("Player One Scores");
  private final JButton playerTwoScores = new JButton("Player Two Scores");
  private final JTextField scoreDisplay = new JTextField(20);
  private final JPanel panel = new JPanel();


  public View() {
    window.setSize(400, 150);
    scoreDisplay.setHorizontalAlignment(JTextField.CENTER);
    scoreDisplay.setEditable(false);
    scoreDisplay.setHorizontalAlignment(JTextField.CENTER);
    scoreDisplay.setEditable(false);

    panel.add(playerOneScores);
    panel.add(playerTwoScores);
    panel.add(scoreDisplay);

    window.add(panel);

    window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    window.setVisible(true);
  }

  public void playerOneScoresAddListener(ActionListener controller1) {
    playerOneScores.addActionListener(controller1);
  }
  void playerTwoScoresAddListener(ActionListener controller2) {
    playerTwoScores.addActionListener(controller2);
  }

  @Override
  public void update(String score, boolean gameEnd) {
    scoreDisplay.setText(score);
    if (gameEnd) {
      playerOneScores.setEnabled(false);
      playerTwoScores.setEnabled(false);
    }
  }
}
