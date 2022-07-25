#include <Arduino.h>
#include "CommandProcessor.h"
#include <SoftwareSerial.h>

#define WIFIPIN 4

extern SoftwareSerial mySerial;

void makeMove(String move)
{
    Serial2.println(move);
    Serial.println(move);

    digitalWrite(WIFIPIN, HIGH);
    delay(250);

    mySerial.println(move);

    delay(250);
    digitalWrite(WIFIPIN, LOW);
}


void process_command(char move)
{
    switch (move)
    {
    case 'r':
        makeMove("R1");
        break;

    case 'f':
        makeMove("F1");
        break;

    case 'b':
        makeMove("B1");
        break;

    case 'l':
        makeMove("L1");
        break;

    case 'u':
        makeMove("U1");
        break;

    case 'd':
        makeMove("D1");
        break;

    default:
        Serial.println("Unknown");
        break;
    }
}

