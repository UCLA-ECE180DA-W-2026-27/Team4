#include <Wire.h>

#define LSM6DSL_ADDR 0x6A
#define LSM6DSL_CTRL1_XL 0x10
#define LSM6DSL_OUTX_L_XL 0x28

void setup() {
  Serial.begin(115200);
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
  Wire.requestFrom(LSM6DSL_ADDR, 2);

  if (Wire.available() >= 2) {
    byte x_low = Wire.read();
    byte x_high = Wire.read();
    int16_t rawX = (int16_t)(x_high << 8 | x_low);

  
    if (rawX < -5000) {
      Serial.println("TILT_LEFT_HIGH");
    } else if (rawX > 5000) {
      Serial.println("TILT_RIGHT_HIGH");
    } else {
      Serial.println("IDLE_LOW");
    }
  } else {
    Serial.println("RETRY_I2C");
  }

  delay(100); 
}