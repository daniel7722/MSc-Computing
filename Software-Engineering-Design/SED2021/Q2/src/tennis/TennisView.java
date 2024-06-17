package tennis;

import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.WindowConstants;

public class TennisView implements Updatable {
  private JTextField scoreDisplay = new JTextField(20);
  private JButton playerOneScores = new JButton("Player One Scores");
  private JButton playerTwoScores = new JButton("Player Two Scores");

  public TennisView () {
    JFrame window = new JFrame("Tennis");
    window.setSize(400, 150);

    scoreDisplay.setHorizontalAlignment(JTextField.CENTER);
    scoreDisplay.setEditable(false);

    JPanel panel = new JPanel();
    panel.add(playerOneScores);
    panel.add(playerTwoScores);
    panel.add(scoreDisplay);

    window.add(panel);

    window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    window.setVisible(true);
  }


  public void playerTwoAddController(ActionListener controller) {
    playerTwoScores.addActionListener(controller);
  }

  public void playerOneAddController(ActionListener controller) {
    playerOneScores.addActionListener(controller);
  }

  @Override
  public void update(String output, boolean gameEnded) {
    scoreDisplay.setText(output);
    if (gameEnded) {
      playerOneScores.setEnabled(false);
      playerTwoScores.setEnabled(false);
    }
  }
}
