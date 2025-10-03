const int RELAY1 = 16;
const int RELAY2 = 17;

void setup() {
  Serial.begin(115200);  // Open serial link at 115200 baud
  pinMode(RELAY1, OUTPUT);
  pinMode(RELAY2, OUTPUT);
  
  // Turn all relays OFF at startup
  digitalWrite(RELAY1, LOW);
  digitalWrite(RELAY2, LOW);

}

void handelCommand(String cmd){
  if (cmd == "L1:ON"){
    digitalWrite(RELAY1, HIGH);
    Serial.println("ACK");
  }
  else if (cmd == "L1:OFF"){
    digitalWrite(RELAY1, LOW);
    Serial.println("ACK");
  }
  else if (cmd == "L2:ON"){
    digitalWrite(RELAY2, HIGH);
    Serial.println("ACK");
  }
  else if (cmd == "L2:OFF"){
    digitalWrite(RELAY2, OFF);
    Serial.println("ACK");
  }
  else if (cmd == "ALL:ON") { digitalWrite(RELAY1, ON); digitalWrite(RELAY2, ON); Serial.println("ACK"); }
  else if (cmd == "ALL:OFF") { digitalWrite(RELAY1, LOW); digitalWrite(RELAY2, LOW); Serial.println("ACK"); }
  else {Serial.println("NACK")}
}

void loop() {
  static String buffer = "";
  
  while (Serial.available()) {
    char c = Serial.read()
    if (c == "\n") {
      handleCommand(buffer);
      buffer == "";
    }
    else buffer += c;
  }

}
