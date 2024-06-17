package ic.doc.camera;

public class Camera implements WriteListener {

  private final Sensor sensor;
  private final MemoryCard memoryCard;
  private boolean isOn;
  private boolean isWriting;

  private boolean sensorOn;

  public Camera(Sensor sensor, MemoryCard memoryCard) {
    this.sensor = sensor;
    this.memoryCard = memoryCard;
    isOn = false;
    isWriting = false;
    sensorOn = false;
  }

  public void pressShutter() {
    if (isOn) {
      memoryCard.write(sensor.readData());
      isWriting = true;
    }
  }

  public void powerOn() {
    isOn = true;
    sensor.powerUp();
    sensorOn = true;
  }

  public void powerOff() {
    isOn = false;
    if (!isWriting) {
      if (sensorOn) {
        sensor.powerDown();
        sensorOn = false;
      }
    }
  }

  @Override
  public void writeComplete() {
    isWriting = false;
    sensor.powerDown();
    sensorOn = false;
  }
}

