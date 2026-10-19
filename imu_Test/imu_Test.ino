#include <Wire.h>
#include <SoftwareSerial.h>

SoftwareSerial BTserial(2, 3); // tx | rx in HC-05 to arduino

#define LSM6DSL_ADDR 0x6A
#define LSM6DSL_CTRL1_XL 0x10
#define LSM6DSL_OUTX_L_XL 0x28

void setup() {
  Serial.begin(115200);
  delay(2000);
  BTserial.begin(9600);
  Wire.begin();
  delay(100);

  Wire.beginTransmission(LSM6DSL_ADDR);
  Wire.write(LSM6DSL_CTRL1_XL);
  Wire.write(0x40); 
  Wire.endTransmission();

  Serial.println("SYSTEM_READY");
}

void loop() {
  Wire.beginTransmission(LSM6DSL_ADDR);
  Wire.write(LSM6DSL_OUTX_L_XL);
  Wire.endTransmission(false);
  
  Wire.requestFrom(LSM6DSL_ADDR, 6); // Read x,y,z axes

  if (Wire.available() >= 6) {
    byte x_low = Wire.read();
    byte x_high = Wire.read();
    byte y_low = Wire.read();
    byte y_high = Wire.read();
    byte z_low = Wire.read();
    byte z_high = Wire.read();

    int16_t rawX = (int16_t)(x_high << 8 | x_low);
    int16_t rawY = (int16_t)(y_high << 8 | y_low);

    String tiltState = "IDLE_LOW";
    if (rawX < -5000) {
      tiltState = "TILT_LEFT_HIGH";
    } else if (rawX > 5000) {
      tiltState = "TILT_RIGHT_HIGH";
    }

    // Print to serial monitor/Bluetooth
    Serial.print("RawX: ");
    Serial.print(rawX);
    Serial.print("\tState: ");
    Serial.println(tiltState);

    BTserial.println(tiltState);
  } else {
    Serial.println("RETRY_I2C");
  }

  delay(100);
}