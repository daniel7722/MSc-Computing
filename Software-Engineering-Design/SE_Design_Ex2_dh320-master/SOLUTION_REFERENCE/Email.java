package contacts;

public class Email extends ContactInfo implements TextMessageEnabled {

  protected final String address;

  public Email(String address) {
    this.address = address;
  }

  @Override
  public String contactInfo() {
    return address;
  }

  @Override
  public String contactInfoType() {
    return "email";
  }

  @Override
  public void sendMessage(String msg) {
    System.out.println(msg + ": " + address);
  }

  @Override
  public void sendTextMessage(String msg) {
    System.out.println(msg + ": " + address);
  }
}
