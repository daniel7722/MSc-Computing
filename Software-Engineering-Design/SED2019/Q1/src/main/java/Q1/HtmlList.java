package Q1;

public class HtmlList implements ListFormatter {

    @Override
    public String formatHeader() {
        return "<ul>";
    }

    @Override
    public String formatItem(String item) {
        return "  <li>" + item + "</li>";
    }

    @Override
    public String formatFooter() {
        return "</ul>";
    }
}
