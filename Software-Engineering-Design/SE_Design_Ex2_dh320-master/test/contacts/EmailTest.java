package contacts;

import static org.hamcrest.Matchers.containsString;
import static org.hamcrest.Matchers.not;
import static org.junit.Assert.assertThat;

import java.io.ByteArrayOutputStream;
import org.junit.Test;

public class EmailTest {

  @Test
  public void sendMessage() {
    ByteArrayOutputStream outstream = TestSuiteHelper.replaceSystemOutStreamForTesting();

    String address = "myEmail@domain.com";
    String msg = "Hello";

    Email email = new Email(address);
    email.sendMessage(msg);

    String actualOutput = outstream.toString();

    assertThat(actualOutput, containsString(msg));
    assertThat(actualOutput, containsString(address));
    assertThat(actualOutput, not(containsString("Audio")));
  }
}
