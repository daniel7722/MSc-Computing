package contacts;

/**
 * Task 2: Mobile Phone Implementation
 *
 * A mobile phone can send BOTH text messages AND audio messages.
 * It extends Phone and adds text messaging capability.
 *
 * REQUIREMENTS:
 * - Extends Phone (inherits audio messaging)
 * - Implements TextMessageEnabled (adds text messaging)
 * - Default sendMessage() sends TEXT, not audio (overrides Phone's behavior!)
 *
 * KEY CONCEPT - POLYMORPHISM:
 * Even though MobilePhone IS-A Phone, when you call sendMessage() on a
 * MobilePhone, it should send a TEXT message, not an audio message!
 *
 * See LEARNING_GUIDE.md for detailed instructions.
 */
public class MobilePhone extends Phone implements AudioMessageEnabled, TextMessageEnabled {

  /**
   * Constructor: Initialize the mobile phone with a phone number
   *
   * Since MobilePhone extends Phone, we need to call Phone's constructor.
   *
   * HINT: Use super(phoneNumber) to call the parent constructor
   *
   * @param phoneNumber the phone number for this mobile phone
   */
  public MobilePhone(String phoneNumber) {
    // TODO: Call the parent class (Phone) constructor using super()

  }

  /**
   * Send a text message (SMS)
   *
   * Output format: "messagePhoneNumber"
   * Example: "Hello+434 9434 132"
   *
   * Note: No "Audio" prefix, no spaces, no colons!
   *
   * HINT: The phoneNumber field is inherited from Phone (that's why it's protected!)
   *
   * @param msg the text message to send
   */
  @Override
  public void sendTextMessage(String msg) {
    // TODO: Print the message followed by the phone number (no spaces)
    // Use: System.out.println(...)

  }

  /**
   * Send an audio message (voice call)
   *
   * Mobile phones can still make voice calls!
   * We want to reuse Phone's implementation.
   *
   * HINT: Use super.sendAudioMessage(msg) to call the parent's implementation
   *
   * @param msg the audio message to send
   */
  @Override
  public void sendAudioMessage(Audio msg) {
    // TODO: Call the parent class's sendAudioMessage method
    // This is already implemented in Phone, so just delegate to it

  }

  /**
   * Send a message (default behavior for mobile is TEXT, not audio!)
   *
   * IMPORTANT: This overrides Phone's sendMessage()!
   * For a mobile phone, the default is to send a TEXT message.
   * For a landline (Phone), the default is to send an AUDIO message.
   *
   * This is polymorphism in action!
   *
   * @param msg the message to send
   */
  @Override
  public void sendMessage(String msg) {
    // TODO: Call sendTextMessage() (NOT sendAudioMessage!)
    // Mobile phones send text by default, unlike landlines

  }
}
