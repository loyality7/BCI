/*
 * BCI Stage 2 — Universal LED Controller (Arduino Uno/Nano/Mega & ESP32)
 * ----------------------------------------------------------------------
 * Uses LED_BUILTIN so no external breadboard or resistor is required.
 * Baud rate: 115200
 *   '1' -> LED ON  (Left-hand imagery detected)
 *   '0' -> LED OFF (Right-hand imagery detected)
 */

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  // Quick 3-blink startup check so you know the board is alive
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(150);
    digitalWrite(LED_BUILTIN, LOW);
    delay(150);
  }
  Serial.println("BCI_READY");
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '1') {
      digitalWrite(LED_BUILTIN, HIGH);
      Serial.println("LED:ON");
    } else if (cmd == '0') {
      digitalWrite(LED_BUILTIN, LOW);
      Serial.println("LED:OFF");
    }
  }
}

