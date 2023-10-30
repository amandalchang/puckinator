//Polar to Cartisian coordinates

#include <Stepper.h>

// change this to the number of steps on your motor
#define STEPS 200

// create an instance of the stepper class, specifying
// the number of steps of the motor and the pins it's
// attached to
Stepper stepper(STEPS, 8, 9, 10, 11);

int stepCount = 0

void setup() {
  Serial.begin(9600);
  theta = 0
  r = 1 // full radius of the arm; a constant
}

void loop() {
// step one step:
  myStepper.step(1);
  Serial.print("steps:");
  Serial.println(stepCount);
  stepCount++;
  delay(500);

  //calculate angle from steps
  theta = stepCount * 1.8 // 1.8 degrees per step; 200 steps / revolution
  x = r * cos(theta)
  y = r * sin(theta)
}
