#include <FlexyStepper.h>

// Pin Defs
const int MOTOR_STEP_PIN1 = 5;
const int MOTOR_DIRECTION_PIN1 = 6;

const int MOTOR_STEP_PIN2 = 8;
const int MOTOR_DIRECTION_PIN2 = 9;

const int LIMIT_SWITCH_PIN_1 = 11;
const int LIMIT_SWITCH_PIN_2 = 12;

// const int ESTOP_PIN = 13;

// Create instances of FlexyStepper
FlexyStepper stepper1;

FlexyStepper stepper2;

// declare estop_val
int estop_val;

// misc constants
const int MOTOR_STEPS_PER_REV = 1600;
const int GEARBOX_RATIO = 3;
const int MOTOR_SPEED = 5;         // speed in revolutions per second
const int MOTOR_ACCELERATION = 5;  // acceleration in revs per second squared


// how much serial data we expect before a newline
const unsigned int MAX_INPUT = 50;

// #define DEBUG
#define HOME

void setup() {
  pinMode(2, INPUT_PULLUP);
  // put your setup code here, to run once:
  delay(1000);
  //attachInterrupt(0, Estop, FALLING); // attach interupt to pin 2 (0 parameter)

  stepper1.connectToPins(MOTOR_STEP_PIN1, MOTOR_DIRECTION_PIN1);
  stepper2.connectToPins(MOTOR_STEP_PIN2, MOTOR_DIRECTION_PIN2);

  Serial.begin(115200);

  stepper1.setStepsPerRevolution(MOTOR_STEPS_PER_REV * GEARBOX_RATIO);
  stepper1.setSpeedInRevolutionsPerSecond(MOTOR_SPEED);
  stepper1.setAccelerationInRevolutionsPerSecondPerSecond(MOTOR_ACCELERATION);

  stepper2.setStepsPerRevolution(MOTOR_STEPS_PER_REV * GEARBOX_RATIO);
  stepper2.setSpeedInRevolutionsPerSecond(MOTOR_SPEED);
  stepper2.setAccelerationInRevolutionsPerSecondPerSecond(MOTOR_ACCELERATION);

#ifdef HOME
  if (stepper1.moveToHomeInRevolutions(1, 0.1, 10, LIMIT_SWITCH_PIN_1) != true) {
    while (true) {
      digitalWrite(11, HIGH);
    }
  } else {
    stepper1.setCurrentPositionInRevolutions(0.315);
    stepper1.setTargetPositionInRevolutions(0);
  }

  if (stepper2.moveToHomeInRevolutions(-1, 0.1, 10, LIMIT_SWITCH_PIN_2) != true) {
    while (true) {
      digitalWrite(11, HIGH);
    }
  } else {
    stepper1.setAccelerationInRevolutionsPerSecondPerSecond(2.5);
    stepper2.setCurrentPositionInRevolutions(-0.355);
    stepper2.setTargetPositionInRevolutions(0);
  }
#endif
  //stepper1.setAccelerationInRevolutionsPerSecondPerSecond(5);
  //stepper2.setAccelerationInRevolutionsPerSecondPerSecond(5);
}

// here to process incoming serial data after a terminator received
void process_data(const char *data) {
#ifdef DEBUG
  Serial.println(data);  // Display the received data
#endif

  if (data[0] == 'i' && data[1] == '\0') {  // Check if the data is 'i'
    stepper1.setCurrentPositionInRevolutions(0.0);
    stepper1.setTargetPositionInRevolutions(0.0);

    stepper2.setCurrentPositionInRevolutions(0.0);
    stepper2.setTargetPositionInRevolutions(0.0);
  } else {
    // Tokenize the string and set the positions of the steppers
    char *token = strtok((char *)data, ",");  // Cast to non-const char* for strtok
    if (token != NULL) {
      float firstNumber = atof(token);  // Convert the first token to a float
      float theta_rot = firstNumber / (2.0 * PI);
      if (true)  //(theta_rot <= 0.15 && theta_rot >= -0.125)
      {
        stepper1.setTargetPositionInRevolutions(theta_rot);
#ifdef DEBUG
        Serial.print("Moving stepper one to position: ");
        Serial.println(theta_rot);
#endif
      } else {
#ifdef DEBUG
        Serial.print("Aborted move due to value out of range: ");
        Serial.println(theta_rot);
#endif
      }

      token = strtok(NULL, ",");  // Get the next token
      if (token != NULL) {
        float secondNumber = atof(token);  // Convert the second token to a float
        float phi_rot = secondNumber / (2.0 * PI);
        if (true)  //phi_rot >= -0.17 && phi_rot <= 0.125)
        {
          stepper2.setTargetPositionInRevolutions(phi_rot);
#ifdef DEBUG
          Serial.print("Moving stepper two to position: ");
          Serial.println(phi_rot);
#endif
        } else {
#ifdef DEBUG
          Serial.print("Aborted move due to value out of range: ");
          Serial.println(phi_rot);
#endif
        }
      }
    }
  }
}

void processIncomingByte(const byte inByte) {
  static char input_line[MAX_INPUT];
  static unsigned int input_pos = 0;

  switch (inByte) {
    case '\n':                    // end of text
      input_line[input_pos] = 0;  // terminating null byte

      // terminator reached! process input_line here ...
      process_data(input_line);

      // reset buffer for next time
      input_pos = 0;
      break;

    case '\r':  // discard carriage return
      break;

    default:
      // keep adding if not full ... allow for terminating null byte
      if (input_pos < (MAX_INPUT - 1))
        input_line[input_pos++] = inByte;
      break;

  }  // end of switch

}  // end of processIncomingByte

void Estop() {
  exit(0);
}

void loop() {
  while (Serial.available() > 0) {
    processIncomingByte(Serial.read());
    // read the first byte in the serial buffer
  }

  if (!stepper1.motionComplete()) {
    stepper1.processMovement();  // this call moves the motor
  }

  if (!stepper2.motionComplete()) {
    stepper2.processMovement();  // this call moves the motor
  }

  // estop_val = digitalRead(ESTOP_PIN);
  // if (estop_val == 1);
  //   interuptfunc
}
