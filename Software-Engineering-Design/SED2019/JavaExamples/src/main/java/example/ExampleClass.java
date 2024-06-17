package example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ExampleClass {

  private static final int LIMIT = 50;
  private final String text;
  private final int number;
  private double anotherNumber = 67.898;
  private boolean decision = false;

  public ExampleClass(String text, int number) {
    this.text = text;
    this.number = number;
  }

  public void doSomething() {

    List<String> list = new ArrayList<>();
    Map<String, Integer> map = new HashMap<>();

    list.add("alice");
    list.add("bob");
    list.add(text);

    map.put("charlie", 4);
    map.put("debbie", 3);

    if (decision) {
      anotherNumber = anotherNumber * 1.9;
    }

    if (number > LIMIT) {
      printTheList(list);
    } else {
      printTheMap(map);
    }

    switch (number) {
      case 1: {
        System.out.println("one");
        break;
      }
      case 2: {
        System.out.println("two");
        break;
      }
      default: {
        System.out.println("not one or two");
      }
    }
  }

  private void printTheMap(Map<String, Integer> map) {
    for (String key : map.keySet()) {
      System.out.println(key + " : " + map.get(key));
    }
  }

  private void printTheList(List<String> list) {
    for (String str : list) {
      System.out.println(str);
    }
  }

  public static void main(String[] args) {
    ExampleClass example = new ExampleClass("some text", 1234);
    example.doSomething();
  }
}
