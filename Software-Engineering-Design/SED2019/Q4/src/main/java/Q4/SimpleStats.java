package Q4;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.List;

public class SimpleStats implements Updatable{

  JTextField currentMax = new JTextField(11);
  JTextField currentMean = new JTextField(11);

  private void display() {

    JFrame frame = new JFrame("Simple Stats");
    frame.setSize(250, 350);
    StatCal statCal = new StatCal();
    statCal.addObserver(this);

    JPanel panel = new JPanel();

    panel.add(new JLabel("Max: value "));
    panel.add(currentMax);
    panel.add(new JLabel("Mean: value "));
    panel.add(currentMean);

    addButton(panel, statCal);

    frame.getContentPane().add(panel);

    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    frame.setVisible(true);
  }

  @Override
  public void update(int max, double mean) {
    currentMax.setText(String.valueOf(max));
    currentMean.setText(String.valueOf(mean));
  }

  private void addButton(JPanel panel, StatCal statCal) {
    for (int i = 1; i <= 12; i++) {
      final int n = i;
      JButton button = new JButton(String.valueOf(i));
      button.addActionListener(e -> statCal.add(n));
      panel.add(button);
    }
  }

  public static void main(String[] args) {
    new SimpleStats().display();
  }
}
