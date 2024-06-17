package contacts;

public class Phone extends ContactInfo implements AudioMessageEnabled {

  protected final String phoneNumber;

  public Phone(String phoneNumber) {
    this.phoneNumber = phoneNumber;
  }

  @Override
  public void sendAudioMessage(Audio msg) {
    System.out.println(msg.toString() + phoneNumber);
  }

  @Override
  public String contactInfo() {
    return phoneNumber;
  }

  @Override
  public String contactInfoType() {
    return "phone";
  }

  @Override
  public void sendMessage(String msg) {
    sendAudioMessage(new Audio(msg));
  }
}
