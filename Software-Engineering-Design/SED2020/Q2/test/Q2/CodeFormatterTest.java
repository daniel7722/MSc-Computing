package Q2;

import static org.junit.Assert.assertEquals;

import java.util.List;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class CodeFormatterTest {


  public CodeFormatter codeFormatter = new CodeFormatter(new Formatter() {
    @Override
    public List<String> startOfBlock() {
      return List.of("{");
    }

    @Override
    public WhiteSpace tabsOrSpaces() {
      return WhiteSpace.TWOSPACES;
    }

    @Override
    public String endOfBlock() {
      return "}";
    }
  });

  @Test
  public void doesNotIndentSingleLine() {
    String original = "Single line of code";
    String output = codeFormatter.format(original);
    assertEquals(original, output);
  }

  @Test
  public void indentsBlock() {
    String original = "{" + "\n" + "body" + "\n" + "}";
    String output = codeFormatter.format(original);
    assertEquals("{" + "\n" + "  body" + "\n" + "}", output);
  }

  @Test
  public void trimsOuterWhitespace() {
    String original = "        {" + "\n" + "body" + "\n" + "}     ";
    String output = codeFormatter.format(original);
    assertEquals("{" + "\n" + "  body" + "\n" + "}", output);
  }

  @Test
  public void correctsExcessiveIndent() {
    String original = "{" + "\n" + "       body" + "\n" + "}";
    String output = codeFormatter.format(original);
    assertEquals("{" + "\n" + "  body" + "\n" + "}", output);
  }

  @Test
  public void indentsNestedBlocks() {
    String original =
        "{" + "\n" + "1" + "\n" + "{" + "\n" + "2" + "\n" + "}" + "\n" + "3 " + "\n" + "}";
    String output = codeFormatter.format(original);
    assertEquals(String.join("\n", "{", "  1", "  {", "    2", "  }", "  3", "}"), output);
  }
}
