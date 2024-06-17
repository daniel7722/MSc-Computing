package ic.doc.camera;

import org.jmock.Expectations;
import org.jmock.integration.junit4.JUnitRuleMockery;
import org.junit.Rule;
import org.junit.Test;

public class CameraTest {

  @Rule
  public JUnitRuleMockery context = new JUnitRuleMockery();

  static final byte[] DATA = new byte[0];

  Sensor sensor = context.mock(Sensor.class);
  MemoryCard memoryCard = context.mock(MemoryCard.class);
  Camera camera = new Camera(sensor, memoryCard);


  @Test
  public void switchingTheCameraOnPowersUpTheSensor() {
    context.checking(new Expectations() {{
      exactly(1).of(sensor).powerUp();
    }});

    camera.powerOn();
  }

  @Test
  public void switchingTheCameraOffOnPowersDownTheSensor() {
    turnOnCamera();
    context.checking(new Expectations() {{
      exactly(1).of(sensor).powerDown();
    }});

    camera.powerOff();
  }

  @Test
  public void pressingTheShutterWhenThePowerIsOffDoesNothing() {
    context.checking(new Expectations() {{
      never(memoryCard);
      never(sensor);
    }});

    camera.pressShutter();
  }

  @Test
  public void pressingTheShutterWithThePowerOnCopiesDataFromTheSensorToTheMemoryCard() {
    turnOnCamera();
    context.checking(new Expectations() {{
      exactly(1).of(sensor).readData();
      will(returnValue(DATA));
      exactly(1).of(memoryCard).write(DATA);
    }});

    camera.pressShutter();
  }

  @Test
  public void switchingCameraOffDuringDataWriteDoesNotPowerDownSensor() {
    turnOnCamera();
    takePicture();
    context.checking(new Expectations() {{
      never(sensor);
    }});

    camera.powerOff();
  }

  @Test
  public void switchingCameraOffAfterDataWriteDoesPowerDownSensor() {
    turnOnCamera();
    takePicture();
    context.checking(new Expectations() {{
      exactly(1).of(sensor).powerDown();
    }});

    camera.writeComplete();
    camera.powerOff();
  }

  @Test
  public void cameraPowersDownSensorIfWritingIsCompleted() {
    turnOnCamera();
    takePicture();
    context.checking(new Expectations() {{
      exactly(1).of(sensor).powerDown();
    }});

    camera.writeComplete();
  }

  private void turnOnCamera() {
    context.checking(new Expectations() {{
      allowing(sensor).powerUp();
    }});

    camera.powerOn();
  }

  private void takePicture() {
    context.checking(new Expectations() {{
      allowing(sensor).readData();
      will(returnValue(DATA));
      allowing(memoryCard).write(DATA);
    }});

    camera.pressShutter();
  }


}
