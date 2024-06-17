package Q2;

import java.util.List;

public interface Formatter {

  public List<String> startOfBlock();

  public WhiteSpace tabsOrSpaces();

  public String endOfBlock();
}
