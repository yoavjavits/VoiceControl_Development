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
    SendCommandToFireBase("Right", "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
}

void MakeForwardMove()
{
    Serial2.println("F1");
    SendCommandToFireBase("Foward", "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
}

void MakeBackwardMove()
{
    Serial2.println("B1");
    SendCommandToFireBase("Backward", "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
}

void MakeLeftMove()
{
    Serial2.println("L1");
    SendCommandToFireBase("Left", "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
}

void MakeUpMove()
{
    Serial2.println("U1");
    SendCommandToFireBase("Up", "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
}

void MakeDownMove()
{
    Serial2.println("D1");
    SendCommandToFireBase("Down", "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB");
}

void SendCommandToFireBase(String move, String state)
{
    if (Firebase.ready() && signupOK)
    {
        String path = "Rotation/Direction/" + String(count);
        if (!Firebase.RTDB.setString(&fbdo, path, move))
        {
            Serial.println("FAILED");
            Serial.println("REASON: " + fbdo.errorReason());
        }
        if (!Firebase.RTDB.setString(&fbdo, "state_cube", state))
        {
            Serial.println("FAILED");
            Serial.println("REASON: " + fbdo.errorReason());
        }
        count++;
    }
}