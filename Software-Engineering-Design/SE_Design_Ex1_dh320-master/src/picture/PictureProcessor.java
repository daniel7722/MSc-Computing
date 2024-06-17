package picture;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class PictureProcessor {
  public static void grayscale(String in, String out) {
    Picture pictureIn = new Picture(in);
    int pictureInWidth = pictureIn.getWidth();
    int pictureInHeight = pictureIn.getHeight();
    Picture pictureOut = new Picture(pictureInWidth, pictureInHeight);

    for (int r = 0; r < pictureInHeight; r++) {
      for (int c = 0; c < pictureInWidth; c++) {
        Color colour = pictureIn.getPixel(c, r);
        int average = (colour.blue() + colour.green() + colour.red()) / 3;
        Color newColour = new Color(average, average, average);
        pictureOut.setPixel(c, r, newColour);
      }
    }
    pictureOut.saveAs(out);
  }

  public static void rotate(String angle, String in, String out) {
    Picture pictureIn = new Picture(in);
    int pictureInWidth = pictureIn.getWidth();
    int pictureInHeight = pictureIn.getHeight();

    if (Objects.equals(angle, "180")) {
      Picture pictureOut = new Picture(pictureInWidth, pictureInHeight);
      for (int r = 0; r < pictureInHeight; r++) {
        for (int c = 0; c < pictureInWidth; c++) {
          pictureOut.setPixel(
              pictureInWidth - c - 1, pictureInHeight - r - 1, pictureIn.getPixel(c, r));
        }
      }
      pictureOut.saveAs(out);
    } else if (Objects.equals(angle, "90")) {
      Picture pictureOut = new Picture(pictureInHeight, pictureInWidth);
      for (int r = 0; r < pictureInHeight; r++) {
        for (int c = 0; c < pictureInWidth; c++) {
          pictureOut.setPixel(pictureInHeight - r - 1, c, pictureIn.getPixel(c, r));
        }
      }
      pictureOut.saveAs(out);
    } else {
      Picture pictureOut = new Picture(pictureInHeight, pictureInWidth);
      for (int r = 0; r < pictureInHeight; r++) {
        for (int c = 0; c < pictureInWidth; c++) {
          pictureOut.setPixel(r, pictureInWidth - c - 1, pictureIn.getPixel(c, r));
        }
      }
      pictureOut.saveAs(out);
    }
  }

  public static void invert(String in, String out) {
    Picture pictureIn = new Picture(in);
    int pictureInWidth = pictureIn.getWidth();
    int pictureInHeight = pictureIn.getHeight();
    Picture pictureOut = new Picture(pictureInWidth, pictureInHeight);

    for (int r = 0; r < pictureInHeight; r++) {
      for (int c = 0; c < pictureInWidth; c++) {
        Color colour = pictureIn.getPixel(c, r);
        Color newColour = new Color(255 - colour.red(), 255 - colour.green(), 255 - colour.blue());
        pictureOut.setPixel(c, r, newColour);
      }
    }
    pictureOut.saveAs(out);
  }

  public static void flip(String dir, String in, String out) {
    Picture pictureIn = new Picture(in);
    int pictureInWidth = pictureIn.getWidth();
    int pictureInHeight = pictureIn.getHeight();
    Picture pictureOut = new Picture(pictureInWidth, pictureInHeight);

    for (int r = 0; r < pictureInHeight; r++) {
      for (int c = 0; c < pictureInWidth; c++) {
        if (Objects.equals(dir, "H")) {
          pictureOut.setPixel(pictureInWidth - c - 1, r, pictureIn.getPixel(c, r));
        } else {
          pictureOut.setPixel(c, pictureInHeight - r - 1, pictureIn.getPixel(c, r));
        }
      }
    }
    pictureOut.saveAs(out);
  }

  public static void blend(List<String> variables) {
    int size = variables.size();
    int minHeight = Integer.MAX_VALUE;
    int minWidth = Integer.MAX_VALUE;
    for (int i = 0; i < size - 1; i++) {
      Picture pictureIn = new Picture(variables.get(i));
      if (pictureIn.getWidth() < minWidth) {
        minWidth = pictureIn.getWidth();
      }
      if (pictureIn.getHeight() < minHeight) {
        minHeight = pictureIn.getHeight();
      }
    }

    Picture pictureOut = new Picture(minWidth, minHeight);

    for (int r = 0; r < minHeight; r++) {
      for (int c = 0; c < minWidth; c++) {
        int red = 0;
        int green = 0;
        int blue = 0;
        for (int i = 0; i < size - 1; i++) {
          Picture pictureIn = new Picture(variables.get(i));
          Color colour = pictureIn.getPixel(c, r);
          red += colour.red();
          blue += colour.blue();
          green += colour.green();
        }
        red /= size - 1;
        blue /= size - 1;
        green /= size - 1;
        Color newColour = new Color(red, green, blue);
        pictureOut.setPixel(c, r, newColour);
      }
    }
    pictureOut.saveAs(variables.get(size - 1));
  }

  public static void neighbour_average(
      int c, int r, Picture pictureOut, int height, int width, Picture pictureIn) {
    int iterateR = r - 1;
    int iterateC = c - 1;
    int red = 0;
    int green = 0;
    int blue = 0;
    boolean boundry = false;
    for (int countRow = 0; countRow < 3; countRow++) {
      for (int countCol = 0; countCol < 3; countCol++) {
        if (iterateR + countRow >= 0
            && iterateR + countRow < height
            && iterateC + countCol >= 0
            && iterateC + countCol < width) {
          Color colour = pictureIn.getPixel(iterateC + countCol, iterateR + countRow);
          red += colour.red();
          green += colour.green();
          blue += colour.blue();
        } else {
          boundry = true;
        }
      }
    }
    Color newColour;
    if (boundry) {
      newColour = pictureIn.getPixel(c, r);
    } else {
      newColour = new Color(red / 9, green / 9, blue / 9);
    }
    pictureOut.setPixel(c, r, newColour);
  }

  public static void blur(String in, String out) {
    Picture pictureIn = new Picture(in);
    int pictureInWidth = pictureIn.getWidth();
    int pictureInHeight = pictureIn.getHeight();
    Picture pictureOut = new Picture(pictureInWidth, pictureInHeight);

    for (int r = 0; r < pictureInHeight; r++) {
      for (int c = 0; c < pictureInWidth; c++) {
        neighbour_average(c, r, pictureOut, pictureInHeight, pictureInWidth, pictureIn);
      }
    }
    pictureOut.saveAs(out);
  }

  public static void mosaic(List<String> variables) {
    int size = variables.size();
    int minHeight = Integer.MAX_VALUE;
    int minWidth = Integer.MAX_VALUE;
    ArrayList<Integer> indexV = new ArrayList<>();

    for (int i = 1; i < (size - 1); i++) {
      Picture pictureIn = new Picture(variables.get(i));
      if (pictureIn.getWidth() < minWidth) {
        minWidth = pictureIn.getWidth();
      }
      if (pictureIn.getHeight() < minHeight) {
        minHeight = pictureIn.getHeight();
      }
      indexV.add(i - 1);
    }
    Picture pictureOut = new Picture(minWidth, minHeight);

    int indd = 0;
    int indd2 = 0;
    for (int r = 0; r < minHeight; r++) {
      indd2 = r / Integer.parseInt(variables.get(0));
      if (indd2 != indd) {
        indexV.add(indexV.remove(0));
        indd = indd2;
      }
      for (int c = 0; c < minWidth; c++) {
        int ind = c / Integer.parseInt(variables.get(0));
        ind = ind % (size - 2);
        Picture p = new Picture(variables.get(indexV.get(ind) + 1));
        pictureOut.setPixel(c, r, p.getPixel(c, r));
      }
    }
    pictureOut.saveAs(variables.get(size - 1));
  }

  public static void main(String[] args) {
    String[] angle = new String[] {"90", "180", "270"};
    String[] direction = new String[] {"H", "V"};
    if (args.length == 0) {
      System.out.println("To get help, write PictureProcessor help");
    } else if (Objects.equals(args[0], "help")) {
      System.out.println("***** Command-Line Options *****");
      System.out.println("help                                     - displays this help menu");
      System.out.println(
          "grayscale <in> <out>                     - write to <out> a monochrome version <in>");
      System.out.println(
          "rotate 90|180|270| <in> <out>            - writes to <out> a rotated version of <in>");
      System.out.println(
          "invert <in> <out>                        - writes to <out> a inverted version of <in>");
      System.out.println(
          "flip [H|V] <in> <out>                    - writes to <out> a flipped either "
              + "Horizontally or Vertically version of <in>");
      System.out.println(
          "blend <in_1> <in_2> ... <out>            - writes to <out> a blended version of all "
              + "inputs");
      System.out.println(
          "mosaic tile-size <in_1> <in_2> ... <out> - writes to <out> a mosaic with tile size "
              + "specified, alternating pattern between each input");
    } else if (Objects.equals(args[0], "grayscale") && args.length == 3) {
      grayscale(args[1], args[2]);
    } else if (Objects.equals(args[0], "rotate")
        && args.length == 4
        && Arrays.asList(angle).contains(args[1])) {
      rotate(args[1], args[2], args[3]);
    } else if (Objects.equals(args[0], "invert") && args.length == 3) {
      invert(args[1], args[2]);
    } else if (Objects.equals(args[0], "flip")
        && args.length == 4
        && Arrays.asList(direction).contains(args[1])) {
      flip(args[1], args[2], args[3]);
    } else if (Objects.equals(args[0], "blend") && args.length >= 3) {
      List<String> variables = new ArrayList<String>(Arrays.asList(args).subList(1, args.length));
      blend(variables);
    } else if (Objects.equals(args[0], "blur") && args.length == 3) {
      blur(args[1], args[2]);
    } else if (Objects.equals(args[0], "mosaic") && args.length >= 4) {
      List<String> variables = new ArrayList<String>(Arrays.asList(args).subList(1, args.length));
      mosaic(variables);
    } else {
      System.out.println("no");
    }
  }
}
