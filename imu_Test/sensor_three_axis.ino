#include <SoftwareSerial.h>
#include <Wire.h>

SoftwareSerial BTserial(2, 3); 
const int LSM6DSL_ADDRESS = 0x6A;

void setup() {
  Serial.begin(9600);
  BTserial.begin(9600);
  Wire.begin();

  #if defined(ARDUINO_ARCH_AVR)
    Wire.setWireTimeout(3000, true); 
  #endif

  delay(500);

  Wire.beginTransmission(LSM6DSL_ADDRESS);
  Wire.write(0x10);  
  Wire.write(0x60);  
  Wire.endTransmission(true);
}

void loop() {
  #if defined(ARDUINO_ARCH_AVR)
    if (Wire.getWireTimeoutFlag()) {
      Wire.clearWireTimeoutFlag();
      return;
    }
  #endif

  Wire.beginTransmission(LSM6DSL_ADDRESS);
  Wire.write(0x28);  
  byte error = Wire.endTransmission(false); 

  if (error == 0) {
    Wire.requestFrom(LSM6DSL_ADDRESS, 6);

    if (Wire.available() >= 6) {
      int16_t x = Wire.read() | (Wire.read() << 8);
      int16_t y = Wire.read() | (Wire.read() << 8);
      int16_t z = Wire.read() | (Wire.read() << 8);

      BTserial.print(x);
      BTserial.print(",");
      BTserial.print(y);
      BTserial.print(",");
      BTserial.println(z);
    }
  }

  delay(50); 
}