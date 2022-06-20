#include <Arduino.h>
#include "CommandProcessor.h"

void process_command(char move)
{
    switch (move)
    {
    case 'r':
        MakeRightMove();
        break;

    case 'f':
        MakeForwardMove();
        break;

    case 'b':
        MakeBackwardMove();
        break;

    case 'l':
        MakeLeftMove();
        break;

    case 'u':
        MakeUpMove();
        break;

    case 'd':
        MakeDownMove();
        break;

    default:
        Serial.println("Unknown");
        break;
    }
}

void MakeRightMove()
{
    Serial2.println("R1");
    Serial2.println("R1");

    Serial.println("R1");
}

void MakeForwardMove()
{
    Serial2.println("F1");
    Serial2.println("F1");

    Serial.println("F1");
}

void MakeBackwardMove()
{
    Serial2.println("B1");
    Serial2.println("B1");

    Serial.println("B1");
}

void MakeLeftMove()
{
    Serial2.println("L1");
    Serial2.println("L1");

    Serial.println("L1");
}

void MakeUpMove()
{
    Serial2.println("U1");
    Serial2.println("U1");

    Serial.println("U1");
}

void MakeDownMove()
{
    Serial2.println("D1");
    Serial2.println("D1");

    Serial.println("D1");
}

void ProcessLearnCommand()
{
    Serial2.println("S1");
    Serial2.println("S1");

    Serial.println("S1");
}