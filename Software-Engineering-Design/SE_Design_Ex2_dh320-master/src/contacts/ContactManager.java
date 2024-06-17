package contacts;

import java.util.ArrayList;
import java.util.List;

public class ContactManager {

  private final List<Contact> contacts = new ArrayList<>();

  public ContactManager() {}

  public void add(Person p, ContactInfo c) {
    Contact contact = new Contact(p, c);
    contacts.add(contact);
  }

  public List<ContactInfo> contactDetails(Person p) {
    List<ContactInfo> contactInfos = new ArrayList<>();
    for (Contact c : contacts) {
      if (c.getP() == p) {
        contactInfos.add(c.getC());
      }
    }
    return contactInfos;
  }

  public void spam(String msg) {
    for (Contact c : contacts) {
      c.getC().sendMessage(msg);
    }
  }
}
