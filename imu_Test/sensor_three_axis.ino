#include <SoftwareSerial.h>
#include <Wire.h>

SoftwareSerial BTserial(2, 3); // RX, TX
const uint8_t IMU_ADDR = 0x6A;
uint8_t zeroCount = 0;

bool writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission(true) == 0;
}

bool initIMU() {
  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x0F);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)IMU_ADDR, 1) != 1) return false;
  if (Wire.read() != 0x6A) return false;

  bool ok = writeReg(0x10, 0x60); 
  ok = writeReg(0x11, 0x60) && ok; 
  return ok;
}

int16_t readInt16() {
  uint8_t lo = Wire.read();
  uint8_t hi = Wire.read();
  return (int16_t)((hi << 8) | lo);
}

void setup() {
  Serial.begin(9600);
  BTserial.begin(9600);
  Wire.begin();
#if defined(ARDUINO_ARCH_AVR)
  Wire.setWireTimeout(3000, true);
#endif
  delay(500);

  while (!initIMU()) {
    Serial.println("IMU init failed, retrying...");
    delay(500);
  }
  Serial.println("IMU OK");
}

void loop() {
#if defined(ARDUINO_ARCH_AVR)
  if (Wire.getWireTimeoutFlag()) {
    Wire.clearWireTimeoutFlag();
    return;
  }
#endif

  Wire.beginTransmission(IMU_ADDR);
  Wire.write(0x22); 
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom((int)IMU_ADDR, 12);
    if (Wire.available() >= 12) {               
      int16_t gx = readInt16();
      int16_t gy = readInt16();
      int16_t gz = readInt16();
      int16_t ax = readInt16();
      int16_t ay = readInt16();
      int16_t az = readInt16();

      float amag = sqrt((float)ax * ax + (float)ay * ay + (float)az * az);
      int16_t yawRate = (amag > 0)
      ? (int16_t)(((float)gx * ax + (float)gy * ay + (float)gz * az) / amag)
      : gz;

      if (ax == 0 && ay == 0 && az == 0) {
        if (++zeroCount >= 10) {
          initIMU();
          zeroCount = 0;
        }
      } else {
        zeroCount = 0;
        char buf[48];
        snprintf(buf, sizeof(buf), "%d,%d,%d,%d", yawRate, ax, ay, az);
        Serial.println(buf);    // USB
        BTserial.println(buf);  // HC-05
      }
    }
  }
  delay(50);
}