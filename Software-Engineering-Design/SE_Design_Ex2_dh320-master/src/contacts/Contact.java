package contacts;

public class Contact {

  private final Person person;
  private final ContactInfo info;

  public Contact(Person person, ContactInfo info) {
    this.person = person;
    this.info = info;
  }

  public Person getP() {
    return person;
  }

  public ContactInfo getC() {
    return info;
  }
}
