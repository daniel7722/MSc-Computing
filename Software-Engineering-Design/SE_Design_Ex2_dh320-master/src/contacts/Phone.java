package contacts;

/**
 * Task 1: Landline Phone Implementation
 *
 * A landline phone can only send audio messages (voice calls).
 *
 * REQUIREMENTS:
 * - Extends ContactInfo (abstract class)
 * - Implements AudioMessageEnabled (interface)
 * - Has a phone number
 * - Can send audio messages
 *
 * See LEARNING_GUIDE.md for detailed instructions.
 */
public class Phone extends ContactInfo implements AudioMessageEnabled {

  // TODO: Declare a protected String field called 'phoneNumber'
  // Why protected? So that MobilePhone (which extends this class) can access it


  /**
   * Constructor: Initialize the phone with a phone number
   *
   * @param phoneNumber the phone number for this phone
   */
  public Phone(String phoneNumber) {
    // TODO: Initialize the phoneNumber field

  }

  /**
   * Send an audio message (voice call)
   *
   * Output format: "contacts.Audio saying 'message'phoneNumber"
   * Example: "contacts.Audio saying 'Hello'+114 53 132"
   *
   * HINT: Use Audio's toString() method and concatenate with phoneNumber
   *
   * @param msg the audio message to send
   */
  @Override
  public void sendAudioMessage(Audio msg) {
    // TODO: Print the audio message followed by the phone number (no spaces)
    // Use: System.out.println(...)

  }

  /**
   * Get the contact information (the phone number)
   *
   * @return the phone number
   */
  @Override
  public String contactInfo() {
    // TODO: Return the phone number
    return null;
  }

  /**
   * Get the type of contact info
   *
   * @return "phone"
   */
  @Override
  public String contactInfoType() {
    // TODO: Return the string "phone"
    return null;
  }

  /**
   * Send a message (default behavior for landline is to send audio)
   *
   * For a landline phone, all messages are voice calls (audio messages).
   *
   * HINT: Create a new Audio object from the message string,
   *       then call sendAudioMessage()
   *
   * @param msg the message to send
   */
  @Override
  public void sendMessage(String msg) {
    // TODO: Convert the string message to an Audio object and send it
    // Step 1: Create new Audio object: new Audio(msg)
    // Step 2: Call sendAudioMessage() with that Audio object

  }
}
