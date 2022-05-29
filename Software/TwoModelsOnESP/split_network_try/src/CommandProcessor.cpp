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