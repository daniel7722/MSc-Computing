package Q2;

import java.util.List;

public class RubyCodeFormatter implements Formatter {
  
  public List<String> startOfBlock() {
    return List.of("do", "if", "while");
  }
  
  public String endOfBlock() {
    return "end";
  }
  
  public WhiteSpace tabsOrSpaces() {
    return WhiteSpace.TABS;
  }
}
