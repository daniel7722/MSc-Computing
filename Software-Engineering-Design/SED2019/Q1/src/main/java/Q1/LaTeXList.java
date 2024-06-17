package Q1;

public class LaTeXList implements ListFormatter {

    @Override
    public String formatHeader() {
        return "\\begin{itemize}";
    }

    @Override
    public String formatItem(String item) {
        return "  \\item " + item;
    }

    @Override
    public String formatFooter() {
        return "\\end{itemize}";
    }
}
