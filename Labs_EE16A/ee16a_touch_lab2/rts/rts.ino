// defining the pin mapping
#define Ypos A0
#define Xpos A1
#define Xneg A2
#define Yneg A3

// global variables
float x,y;
boolean touched;

/* setup() gets called once at the very beginning.
Responsible for initializing necessary variables/processes. */
void setup()
{
  Serial.begin(9600); // initializes the Serial Monitor
  setup_blink(); // see setup_blink() below
}
/* the MSP430 calls loop() repeatedly until it loses power.
You can think of this function as being stuck inside an 
implicit while(true) loop. */
void loop()
{
  if(touch()) { // if touch() is true, a touch has been detected
    if (!touched) {
      touched = true;
    }
    digitalWrite(GREEN_LED, HIGH); // turn on the on-board Green LED
    x = get_loc(Xneg, Xpos, Yneg, Ypos); // call to get_loc for x position
    y = get_loc(Yneg, Ypos, Xneg, Xpos); // call to get_loc for y position
    Serial.print("X = ");
    Serial.print(x);
    Serial.print(" Y = ");
    Serial.println(y);
  } 
  else if (touched) {
    touched = false; // resetting touched
  }
  delay(100); // 100 millisecond delay
  digitalWrite(GREEN_LED, LOW);
}

/* touch() returns true if a touch has been detected;
returns false otherwise. You can ignore this function, 
unless you're curious. */
boolean touch()
{
  pinMode(Ypos, INPUT_PULLUP);
  pinMode(Yneg, INPUT);
  pinMode(Xneg, OUTPUT);
  pinMode(Xpos, INPUT);
  digitalWrite(Xneg, LOW);
  boolean touch = false;
  if(!digitalRead(Ypos)) {
     touch = true;
  }
  //Serial.println(touch);
  return touch; 
}

/* get_loc is the heart of this program. Returns the location of the touch
along a specific axis. The axis is determined by the 4 input parameters. 
Recall, which layer do you drive and which layer do you sense to get X position?
How about Y? */
int get_loc(int pwr_neg, int pwr_pos, int sense_neg, int sense_pos) {
  pinMode(sense_neg, INPUT); // sets sense_neg to be an input pin
  pinMode(sense_pos, INPUT); // sets sense_pos to be an input pin
  digitalWrite(sense_neg, LOW); // outputs GND to sense_neg to protect from floating voltages
  pinMode(pwr_neg, OUTPUT); // sets pwr_neg to output
  digitalWrite(pwr_neg, LOW); // outputs GND to pwr_neg
  pinMode(pwr_pos, OUTPUT); // sets pwr_pos to output
  digitalWrite(pwr_pos, HIGH); // outputs +3V to pwr_pos
  return analogRead(sense_pos);
} 

/* Initializes the MSP430 */
void setup_blink() {
  pinMode(RED_LED, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  for(int i = 0; i < 5; i++) { // blink the on-board Red & Green LEDs
    digitalWrite(RED_LED, HIGH);
    delay(200);
    digitalWrite(RED_LED, LOW);
    delay(200);  
  }
  digitalWrite(GREEN_LED, HIGH); // turn on the Green LED for 1 sec.
  delay(1000);
  digitalWrite(GREEN_LED, LOW);
  Serial.println("Touch Screen is Ready...");
}

