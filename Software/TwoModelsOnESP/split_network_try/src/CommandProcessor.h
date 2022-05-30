#ifndef _intent_processor_h_
#define _intent_processor_h_

#include <Firebase_ESP_Client.h>

void process_command(char move);

void MakeRightMove();
void MakeForwardMove();
void MakeBackwardMove();
void MakeLeftMove();
void MakeUpMove();
void MakeDownMove();

void SendCommandToFireBase(String move);

#endif
