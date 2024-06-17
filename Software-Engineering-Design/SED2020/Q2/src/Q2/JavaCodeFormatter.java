package Q2;

import java.util.List;

public class JavaCodeFormatter implements Formatter {
  
  public List<String> startOfBlock() {
    return List.of("{");
  }
  
  public String endOfBlock() {
    return "}";
  }
  public WhiteSpace tabsOrSpaces() {
    return WhiteSpace.TWOSPACES;
  }

}
