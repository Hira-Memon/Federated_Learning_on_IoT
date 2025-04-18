/*
  Weather Prediction for Arduino MKR Vidor 4000
  This code implements a pre-trained decision tree for precipitation prediction
  based on temperature, humidity, and date/time information.
*/

#include <RTCZero.h>

#include <Wire.h>
#include <Adafruit_SHTC3.h>

Adafruit_SHTC3 shtc3;
// Decision tree parameters
const int NUM_NODES = 63;  // For our trained decision tree

// Arrays to store the decision tree structure
// These arrays contain values from your trained model
int feature_indices[NUM_NODES] = {1, 1, 1, 0, 1, -2, -2, 1, -2, -2, 0, 1, -2, -2, 3, -2, -2, 2, 0, 2, -2, -2, 3, -2, -2, 3, 1, -2, -2, 0, -2, -2, 2, 1, 3, 0, -2, -2, 2, -2, -2, 3, 3, -2, -2, 2, -2, -2, 3, 0, 2, -2, -2, 3, -2, -2, 0, 2, -2, -2, 3, -2, -2};

float thresholds[NUM_NODES] = {88.17499923706055, 84.24499893188477, 75.19499969482422, 9.050000190734863, 69.77499771118164, -2.0, -2.0, 67.5999984741211, -2.0, -2.0, 5.1499998569488525, 75.20499801635742, -2.0, -2.0, 127.5, -2.0, -2.0, 10.5, 13.650000095367432, 9.5, -2.0, -2.0, 124.5, -2.0, -2.0, 31.5, 84.42499923706055, -2.0, -2.0, -1.199999988079071, -2.0, -2.0, 8.5, 90.7249984741211, 122.5, 5.700000047683716, -2.0, -2.0, 1.0, -2.0, -2.0, 137.0, 2.5, -2.0, -2.0, 0.5, -2.0, -2.0, 67.0, 3.0, 20.5, -2.0, -2.0, 35.0, -2.0, -2.0, 7.049999952316284, 20.5, -2.0, -2.0, 79.5, -2.0, -2.0};

float values[NUM_NODES][2] = {
  {0.8923130596580998, 0.10768694034190023},
  {0.9560881008449924, 0.04391189915500762},
  {0.9722521967010945, 0.027747803298905502},
  {0.9858247422680413, 0.014175257731958763},
  {0.9948939512961509, 0.005106048703849175},
  {0.9982905982905983, 0.0017094017094017094},
  {0.9873577749683944, 0.012642225031605562},
  {0.9748815165876777, 0.025118483412322274},
  {0.9863325740318907, 0.01366742596810934},
  {0.9180790960451978, 0.08192090395480225},
  {0.9377389404696886, 0.0622610595303113},
  {0.9729944400317713, 0.027005559968228753},
  {0.0, 1.0},
  {0.9737678855325914, 0.026232114467408585},
  {0.8601398601398601, 0.13986013986013987},
  {0.815, 0.185},
  {0.9651162790697675, 0.03488372093023256},
  {0.8128415300546448, 0.1871584699453552},
  {0.8875502008032129, 0.11244979919678715},
  {0.9006928406466512, 0.09930715935334873},
  {0.9113300492610837, 0.08866995073891626},
  {0.7407407407407407, 0.25925925925925924},
  {0.8, 0.2},
  {0.5909090909090909, 0.4090909090909091},
  {0.9069767441860465, 0.09302325581395349},
  {0.6538461538461539, 0.34615384615384615},
  {0.8611111111111112, 0.1388888888888889},
  {0.0, 1.0},
  {0.8985507246376812, 0.10144927536231885},
  {0.5617283950617284, 0.4382716049382716},
  {0.21052631578947367, 0.7894736842105263},
  {0.6083916083916084, 0.3916083916083916},
  {0.558695652173913, 0.44130434782608696},
  {0.6599462365591398, 0.3400537634408602},
  {0.8199233716475096, 0.18007662835249041},
  {0.7897196261682243, 0.2102803738317757},
  {0.8511904761904762, 0.1488095238095238},
  {0.5652173913043478, 0.43478260869565216},
  {0.9574468085106383, 0.0425531914893617},
  {0.5, 0.5},
  {0.9777777777777777, 0.022222222222222223},
  {0.5734989648033126, 0.42650103519668736},
  {0.545045045045045, 0.45495495495495497},
  {0.9583333333333334, 0.041666666666666664},
  {0.5214285714285715, 0.4785714285714286},
  {0.8974358974358975, 0.10256410256410256},
  {0.3333333333333333, 0.6666666666666666},
  {0.9444444444444444, 0.05555555555555555},
  {0.44025157232704404, 0.559748427672956},
  {0.5706371191135734, 0.4293628808864266},
  {0.44387755102040816, 0.5561224489795918},
  {0.3708609271523179, 0.6291390728476821},
  {0.6888888888888889, 0.3111111111111111},
  {0.7212121212121212, 0.2787878787878788},
  {0.8, 0.2},
  {0.5636363636363636, 0.43636363636363634},
  {0.2690909090909091, 0.730909090909091},
  {0.5087719298245614, 0.49122807017543857},
  {0.6341463414634146, 0.36585365853658536},
  {0.1875, 0.8125},
  {0.20642201834862386, 0.7935779816513762},
  {0.0, 1.0},
  {0.2356020942408377, 0.7643979057591623}
};

int children_left[NUM_NODES] = {1, 2, 3, 4, 5, -1, -1, 8, -1, -1, 11, 12, -1, -1, 15, -1, -1, 18, 19, 20, -1, -1, 23, -1, -1, 26, 27, -1, -1, 30, -1, -1, 33, 34, 35, 36, -1, -1, 39, -1, -1, 42, 43, -1, -1, 46, -1, -1, 49, 50, 51, -1, -1, 54, -1, -1, 57, 58, -1, -1, 61, -1, -1};

int children_right[NUM_NODES] = {32, 17, 10, 7, 6, -1, -1, 9, -1, -1, 14, 13, -1, -1, 16, -1, -1, 25, 22, 21, -1, -1, 24, -1, -1, 29, 28, -1, -1, 31, -1, -1, 48, 41, 38, 37, -1, -1, 40, -1, -1, 45, 44, -1, -1, 47, -1, -1, 56, 53, 52, -1, -1, 55, -1, -1, 60, 59, -1, -1, 62, -1, -1};

// Sensor pins
const int temperaturePin = A0;
const int humidityPin = A1;

// RTC for getting date/time
RTCZero rtc;

// Initialize the SHTC3 sensor in your setup() function
void initializeSensors() {
  if (!shtc3.begin()) {
    Serial.println("Couldn't find SHTC3 sensor!");
    while (1) delay(1); // Don't proceed if sensor not found
  }
  Serial.println("SHTC3 sensor initialized successfully");
}

void setup() {
  Serial.begin(9600);
  while (!Serial);
  
  Serial.println("Weather Prediction System");
  Serial.println("Based on trained decision tree model");
  
  // Initialize I2C
  Wire.begin();
  
  // Initialize the SHTC3 sensor
  initializeSensors();
  
  // Initialize RTC
  rtc.begin();
  
  // Set the time (you would typically sync this with a real-time source)
  // Format: hour, minute, second, day, month, year
  rtc.setTime(12, 0, 0);
  rtc.setDate(4, 4, 25);  // April 4, 2025
  
  Serial.println("System ready!");
}

void loop() {
  // Read sensor data
  float temperature = readTemperature();
  float humidity = readHumidity();
  
  // Get current date/time
  int hour = rtc.getHours();
  int month = rtc.getMonth();
  int day = rtc.getDay();
  
  // Calculate day of year (simplified)
  int dayOfYear = calculateDayOfYear(day, month);
  
  // Determine season
  int season = getSeason(month);
  
  // Make prediction using decision tree
  bool willPrecipitate = predictPrecipitation(temperature, humidity, hour, dayOfYear, season);
  
  // Display results
  Serial.println("----------------------------------------");
  Serial.print("Temperature: "); Serial.print(temperature); Serial.println(" °C");
  Serial.print("Humidity: "); Serial.print(humidity); Serial.println(" %");
  Serial.print("Time: "); Serial.print(hour); Serial.println(":00");
  Serial.print("Day of year: "); Serial.println(dayOfYear);
  Serial.print("Season: "); Serial.println(getSeasonName(season));
  Serial.print("Precipitation forecast: ");
  if (willPrecipitate) {
    Serial.println("Rain/Snow Expected");
  } else {
    Serial.println("No Precipitation Expected");
  }
  
  delay(5000);  // Update every minute
}
// Read temperature from SHTC3 sensor
float readTemperature() {
  sensors_event_t humidity, temp;
  shtc3.getEvent(&humidity, &temp);
  
  float temperature = temp.temperature;
  
  // For testing - simulate temperature when sensor reading fails
  if (isnan(temperature)) {
    // Fallback to simulated values
    temperature = 20.0 + random(-50, 50) / 10.0;
    Serial.println("Warning: Using simulated temperature data");
  }
  
  return temperature;
}

// Read humidity from SHTC3 sensor
float readHumidity() {
  sensors_event_t humidity, temp;
  shtc3.getEvent(&humidity, &temp);
  
  float humidityValue = humidity.relative_humidity;
  
  // For testing - simulate humidity when sensor reading fails
  if (isnan(humidityValue)) {
    // Fallback to simulated values
    humidityValue = 70.0 + random(-200, 200) / 10.0;
    Serial.println("Warning: Using simulated humidity data");
  }
  
  return humidityValue;
}

// Calculate day of year from day and month
int calculateDayOfYear(int day, int month) {
  // Simple approximation (not accounting for leap years)
  int daysInMonth[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int dayOfYear = day;
  
  for (int i = 1; i < month; i++) {
    dayOfYear += daysInMonth[i];
  }
  
  return dayOfYear;
}

// Get season from month
int getSeason(int month) {
  if (month == 12 || month == 1 || month == 2) {
    return 1;  // Winter
  } else if (month >= 3 && month <= 5) {
    return 2;  // Spring
  } else if (month >= 6 && month <= 8) {
    return 3;  // Summer
  } else {
    return 4;  // Fall
  }
}

// Get season name for display
String getSeasonName(int season) {
  switch (season) {
    case 1: return "Winter";
    case 2: return "Spring";
    case 3: return "Summer";
    case 4: return "Fall";
    default: return "Unknown";
  }
}

// Predict precipitation using the decision tree
bool predictPrecipitation(float temperature, float humidity, int hour, int dayOfYear, int season) {
  // Create feature array in the same order as training
  // Based on feature_indices values, the mapping is:
  // 0: temperature, 1: humidity, 2: hour, 3: dayOfYear, 4: season
  float features[5] = {temperature, humidity, hour, dayOfYear, season};
  
  // Start at the root node
  int currentNode = 0;
  
  // Debug info
  Serial.println("Decision path:");
  
  // Traverse the tree until reaching a leaf node
  while (true) {
    // Check if it's a leaf node (where feature_indices[currentNode] < 0)
    if (feature_indices[currentNode] < 0) {
      // Leaf node reached, return the prediction
      float precipProb = values[currentNode][1];
      Serial.print("Reached leaf node. Precipitation probability: ");
      Serial.print(precipProb * 100);
      Serial.println("%");
      
      // If precipitation probability > 50%, predict precipitation
      return precipProb > 0.5;
    }
    
    // Get the relevant feature for this node
    int featureIndex = feature_indices[currentNode];
    float featureValue = features[featureIndex];
    
    // Debug info
    Serial.print("Node ");
    Serial.print(currentNode);
    Serial.print(" checking feature ");
    
    // Print feature name for better readability
    switch(featureIndex) {
      case 0: Serial.print("temperature"); break;
      case 1: Serial.print("humidity"); break;
      case 2: Serial.print("hour"); break;
      case 3: Serial.print("dayOfYear"); break;
      case 4: Serial.print("season"); break;
      default: Serial.print("unknown"); break;
    }
    
    Serial.print(" (");
    Serial.print(featureValue);
    Serial.print(") vs threshold ");
    Serial.println(thresholds[currentNode]);
    
    // Compare feature value to threshold and move to the appropriate child
    if (featureValue <= thresholds[currentNode]) {
      Serial.print("Taking left branch to node ");
      Serial.println(children_left[currentNode]);
      currentNode = children_left[currentNode];
    } else {
      Serial.print("Taking right branch to node ");
      Serial.println(children_right[currentNode]);
      currentNode = children_right[currentNode];
    }
  }
}