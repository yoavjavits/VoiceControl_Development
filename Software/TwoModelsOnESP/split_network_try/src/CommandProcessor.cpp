#include <Arduino.h>
#include "CommandProcessor.h"

extern int count;
extern bool signupOK;
extern FirebaseData fbdo;

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
    SendCommandToFireBase("Right");
}

void MakeForwardMove()
{
    Serial2.println("F1");
    SendCommandToFireBase("Foward");
}

void MakeBackwardMove()
{
    Serial2.println("B1");
    SendCommandToFireBase("Backward");
}

void MakeLeftMove()
{
    Serial2.println("L1");
    SendCommandToFireBase("Left");
}

void MakeUpMove()
{
    Serial2.println("U1");
    SendCommandToFireBase("Up");
}

void MakeDownMove()
{
    Serial2.println("D1");
    SendCommandToFireBase("Down");
}

void SendCommandToFireBase(String move)
{
    if (Firebase.ready() && signupOK)
    {
        String path = "Rotation/Direction/" + String(count);
        if (!Firebase.RTDB.setString(&fbdo, "Rotation/Direction", move))
        {
            Serial.println("FAILED");
            Serial.println("REASON: " + fbdo.errorReason());
        }
        count++;
    }
}