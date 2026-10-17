// Color bench firmware (Arduino Uno)
// Protocol: PC sends one letter + newline, board answers with EXACTLY ONE line.
//   I  -> ID COLOR_BENCH
//   D  -> LIGHT <0-1023>   (all LEDs off)
//   R  -> LIGHT <0-1023>   (red LED on)
//   G  -> LIGHT <0-1023>   (green LED on)
//   B  -> LIGHT <0-1023>   (blue LED on)
//   anything else -> ERR unknown_command

const int RED_PIN = 2;
const int GREEN_PIN = 4;
const int BLUE_PIN = 8;
const int SENSOR_PIN = A0;

const int SETTLE_MS = 500;        // wait after switching the LED, same for every command
const int N_SAMPLES = 50;         // readings averaged per measurement
const int SAMPLE_DELAY_MS = 20;

char cmd = 0;                     // first character of the line being received
int len = 0;                      // number of characters in the line so far

void allOff() {
  digitalWrite(RED_PIN, LOW);
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(BLUE_PIN, LOW);
}

unsigned long readAverage() {
  unsigned long sum = 0;          // 50 * 1023 does not fit in an int on the Uno
  for (int i = 0; i < N_SAMPLES; i++) {
    sum += analogRead(SENSOR_PIN);
    delay(SAMPLE_DELAY_MS);
  }
  return sum / N_SAMPLES;
}

// ledPin = -1 means "dark": all LEDs off
void measure(int ledPin) {
  allOff();
  if (ledPin >= 0) digitalWrite(ledPin, HIGH);
  delay(SETTLE_MS);
  unsigned long value = readAverage();
  allOff();
  Serial.print("LIGHT ");
  Serial.println(value);
}

void handleCommand(char c, int n) {
  if (n != 1) {
    Serial.println("ERR unknown_command");
    return;
  }
  switch (c) {
    case 'I': Serial.println("ID COLOR_BENCH"); break;
    case 'D': measure(-1); break;
    case 'R': measure(RED_PIN); break;
    case 'G': measure(GREEN_PIN); break;
    case 'B': measure(BLUE_PIN); break;
    default:  Serial.println("ERR unknown_command"); break;
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(RED_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BLUE_PIN, OUTPUT);
  pinMode(SENSOR_PIN, INPUT);
  allOff();
  Serial.println("READY");
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\r') continue;                    // ignore Windows line endings
    if (c == '\n') {
      if (len > 0) handleCommand(cmd, len);     // ignore empty lines
      cmd = 0;
      len = 0;
    } else {
      if (len == 0) cmd = c;
      len++;
    }
  }
}
