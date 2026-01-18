package contacts;

public class MobilePhone extends Phone implements AudioMessageEnabled, TextMessageEnabled {

  public MobilePhone(String phoneNumber) {
    super(phoneNumber);
  }

  @Override
  public void sendTextMessage(String msg) {
    System.out.println(msg + phoneNumber);
  }

  @Override
  public void sendAudioMessage(Audio msg) {
    super.sendAudioMessage(msg);
  }

  @Override
  public void sendMessage(String msg) {
    sendTextMessage(msg);
  }
}
