package contacts;

import java.util.ArrayList;
import java.util.List;

/**
 * Task 4: Contact Manager Implementation
 *
 * Manages a collection of contacts (Person + ContactInfo pairs).
 *
 * REQUIREMENTS:
 * - Store contacts in an ArrayList
 * - Add new contacts
 * - Retrieve all contact info for a specific person
 * - Send messages to all contacts (spam)
 *
 * KEY CONCEPT - POLYMORPHISM:
 * The spam() method demonstrates polymorphism beautifully!
 * Each ContactInfo type (Phone/MobilePhone/Email) formats messages
 * differently, but ContactManager doesn't need to know which type it is.
 *
 * See LEARNING_GUIDE.md for detailed instructions.
 */
public class ContactManager {

  // TODO: Declare a private final field called 'contacts' of type List<Contact>
  // Initialize it to a new ArrayList<>()
  // Example: private final List<Contact> contacts = new ArrayList<>();


  /**
   * Constructor: Create an empty contact manager
   *
   * Since we're initializing the contacts list at declaration,
   * this constructor can be empty.
   */
  public ContactManager() {
    // TODO: No code needed here if you initialized contacts above
    // But you can also initialize contacts here if you prefer:
    // contacts = new ArrayList<>();
  }

  /**
   * Add a new contact linking a person with their contact information
   *
   * ALGORITHM:
   * 1. Create a new Contact object from the person and contact info
   * 2. Add it to the contacts list
   *
   * HINT: Use the Contact constructor: new Contact(p, c)
   *       Then add it to the list: contacts.add(...)
   *
   * @param p the person
   * @param c the contact information for this person
   */
  public void add(Person p, ContactInfo c) {
    // TODO: Create a Contact and add it to the contacts list
    // Step 1: Contact contact = new Contact(p, c);
    // Step 2: contacts.add(contact);

  }

  /**
   * Get all contact information for a specific person
   *
   * Returns a list of all ContactInfo objects associated with this person.
   * If the person has no contacts, returns an empty list.
   *
   * ALGORITHM:
   * 1. Create an empty ArrayList to store the results
   * 2. Loop through all contacts in the contacts list
   * 3. For each contact, check if the person matches (use == comparison)
   * 4. If it matches, add the ContactInfo to the results list
   * 5. Return the results list
   *
   * HINTS:
   * - Use a for-each loop: for (Contact c : contacts)
   * - Use c.getP() to get the Person from a Contact
   * - Use c.getC() to get the ContactInfo from a Contact
   * - Use == to compare Person objects (reference equality)
   *
   * EXAMPLE:
   * Person alice = new Person("Alice");
   * cm.add(alice, new Email("alice@email.com"));
   * cm.add(alice, new MobilePhone("123-456"));
   * List<ContactInfo> aliceContacts = cm.contactDetails(alice);
   * // Returns a list with 2 items: the Email and the MobilePhone
   *
   * @param p the person to find contact information for
   * @return a list of all ContactInfo for this person (empty if none found)
   */
  public List<ContactInfo> contactDetails(Person p) {
    // TODO: Implement the algorithm described above
    // Step 1: Create empty ArrayList
    // List<ContactInfo> contactInfos = new ArrayList<>();

    // Step 2-4: Loop through contacts and filter by person
    // for (Contact c : contacts) {
    //   if (c.getP() == p) {
    //     contactInfos.add(c.getC());
    //   }
    // }

    // Step 5: Return the list
    // return contactInfos;

    return null; // Replace this with your implementation
  }

  /**
   * Send a message to ALL contacts in the system
   *
   * This is where polymorphism shines! Each contact type will
   * format and send the message differently:
   * - Phone → audio message with "Audio" prefix
   * - MobilePhone → text message without "Audio"
   * - Email → email format with ": " separator
   *
   * The ContactManager doesn't need to know which type each contact is!
   *
   * ALGORITHM:
   * 1. Loop through all contacts
   * 2. For each contact, get the ContactInfo
   * 3. Call sendMessage(msg) on that ContactInfo
   *
   * HINTS:
   * - Use a for-each loop: for (Contact c : contacts)
   * - Use c.getC() to get the ContactInfo
   * - Call .sendMessage(msg) on the ContactInfo
   * - That's it! Polymorphism handles the rest!
   *
   * EXAMPLE OUTPUT (if contacts contain Phone, Email, MobilePhone):
   * contacts.Audio saying 'Hello'+114 53 132
   * Hello: emailP1@address.com
   * Hello+434 9434 132
   *
   * @param msg the message to send to all contacts
   */
  public void spam(String msg) {
    // TODO: Loop through all contacts and send the message
    // for (Contact c : contacts) {
    //   c.getC().sendMessage(msg);
    // }

  }
}
