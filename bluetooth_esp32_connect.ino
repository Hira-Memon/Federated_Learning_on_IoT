#include <BluetoothSerial.h>
#include <ArduinoJson.h>
#include "Adafruit_SHTC3.h"
#include <Wire.h>

#define I2C_SDA 21
#define I2C_SCL 22
#define MAX_READINGS 20  // Keep small to avoid memory issues

BluetoothSerial SerialBT;
Adafruit_SHTC3 shtc3 = Adafruit_SHTC3();

// Simple circular buffer storage
float temperatures[MAX_READINGS];
float humidities[MAX_READINGS];
int dataIndex = 0;

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32_IoT");
  Wire.begin(I2C_SDA, I2C_SCL);
  
  if (!shtc3.begin()) {
    Serial.println("Couldn't find SHTC3");
    while (1) delay(1);
  }
}

void loop() {
  sensors_event_t humidity, temp;
  shtc3.getEvent(&humidity, &temp);
  
  // Store data in circular buffer
  temperatures[dataIndex] = temp.temperature;
  humidities[dataIndex] = humidity.relative_humidity;
  dataIndex = (dataIndex + 1) % MAX_READINGS;
  
  // Same Bluetooth code that works
  StaticJsonDocument<200> doc;
  doc["temperature"] = temp.temperature;
  doc["humidity"] = humidity.relative_humidity;
  
  String jsonString;
  serializeJson(doc, jsonString);
  SerialBT.println(jsonString);
  Serial.println("Sent: " + jsonString);
  delay(2000);
}

// Function to calculate average temperature and humidity
void calculateStats() {
  float sumTemp = 0;
  float sumHum = 0;
  
  for (int i = 0; i < MAX_READINGS; i++) {
    sumTemp += temperatures[i];
    sumHum += humidities[i];
  }
  
  float avgTemp = sumTemp / MAX_READINGS;
  float avgHum = sumHum / MAX_READINGS;
  
  Serial.println("Avg Temp: " + String(avgTemp) + "°C, Avg Humidity: " + String(avgHum) + "%");
}