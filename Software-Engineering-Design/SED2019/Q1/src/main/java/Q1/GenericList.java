package Q1;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GenericList {

    private List<String> content = new ArrayList<>();
    private final ListFormatter listFormatter;

    public GenericList(ListFormatter listFormatter, String... items) {
        this.listFormatter = listFormatter;
        content.addAll(Arrays.asList(items));
    }

    public void add(String item) {
        content.add(item);
    }

  public void print() {
    System.out.println(listFormatter.formatHeader());
    for (String item : content) {
      System.out.println(listFormatter.formatItem(item));
    }
    System.out.println(listFormatter.formatFooter());
    }
}

