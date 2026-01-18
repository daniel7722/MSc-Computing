package contacts;

/**
 * Task 3: Email Implementation
 *
 * An email contact can send text messages (emails).
 *
 * REQUIREMENTS:
 * - Extends ContactInfo (abstract class)
 * - Implements TextMessageEnabled (interface)
 * - Has an email address
 * - Can send text messages
 *
 * Note: Similar structure to Phone, but sends text instead of audio.
 *
 * See LEARNING_GUIDE.md for detailed instructions.
 */
public class Email extends ContactInfo implements TextMessageEnabled {

  // TODO: Declare a protected String field called 'address'
  // This will store the email address


  /**
   * Constructor: Initialize the email with an email address
   *
   * @param address the email address
   */
  public Email(String address) {
    // TODO: Initialize the address field

  }

  /**
   * Get the contact information (the email address)
   *
   * @return the email address
   */
  @Override
  public String contactInfo() {
    // TODO: Return the email address
    return null;
  }

  /**
   * Get the type of contact info
   *
   * @return "email"
   */
  @Override
  public String contactInfoType() {
    // TODO: Return the string "email"
    return null;
  }

  /**
   * Send a message via email
   *
   * For emails, we just send a text message.
   *
   * @param msg the message to send
   */
  @Override
  public void sendMessage(String msg) {
    // TODO: Call sendTextMessage() to send the message
    // Email's sendMessage should behave the same as sendTextMessage

  }

  /**
   * Send a text message (email)
   *
   * Output format: "message: emailAddress"
   * Example: "Hello: emailP1@address.com"
   *
   * Note: There's a colon AND a space between message and address!
   *       This is different from Phone's format.
   *
   * @param msg the text message to send
   */
  @Override
  public void sendTextMessage(String msg) {
    // TODO: Print the message, then ": ", then the email address
    // Use: System.out.println(...)

  }
}
