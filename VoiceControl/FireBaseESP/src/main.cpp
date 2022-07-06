#include <Arduino.h>
#include <WiFi.h>
#include "SPIFFS.h"
#include <Firebase_ESP_Client.h>
#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"
#include <esp_task_wdt.h>
#include "communication_handler.h"
#include <string>

using namespace std;

#define WIFI_SSID "TechPublic"
#define WIFI_PASSWORD ""

#define RXp2 16
#define TXp2 17

#define WIFIPIN 33

#define API_KEY "AIzaSyAonx30i-i5ww0ftTMvQYrkc8eX2J1DOx4"
#define DATABASE_URL "https://voice-control-44c54-default-rtdb.firebaseio.com/"

FirebaseData fbdo;

FirebaseAuth auth;
FirebaseConfig config;

bool signupOK = false;

CommunicatioHandler communication_handler;

void ConnectToFireBase()
{
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(300);
  }

  config.api_key = API_KEY;

  /* Assign the RTDB URL (required) */
  config.database_url = DATABASE_URL;

  /* Sign up */
  if (Firebase.signUp(&config, &auth, "", ""))
  {
    Serial.println("ok");
    signupOK = true;
  }
  else
  {
    Serial.printf("%s\n", config.signer.signupError.message.c_str());
  }

  /* Assign the callback function for the long running token generation task */
  config.token_status_callback = tokenStatusCallback; // see addons/TokenHelper.h

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void setup()
{
  Serial.begin(9600);
  Serial2.begin(115200, SERIAL_8N1, RXp2, TXp2);

  pinMode(WIFIPIN, INPUT);

  // communication_handler.init_communication();

  ConnectToFireBase();

  delay(1000);
}

void SendCommandToFireBase(String move)
{
  if (Firebase.ready() && signupOK)
  {
    String path = (move != "Solve" ? "Rotation/Direction" : "Solve");
    if (!Firebase.RTDB.setString(&fbdo, path, move))
    {
      Serial.println("FAILED");
      Serial.println("REASON: " + fbdo.errorReason());
    }
  }
}

bool execute_comand(char command, int indicator)
{
  bool executed = false;
  unsigned long current_time = millis();
  String to_send = "";
  to_send += command;

  switch (indicator)
  {
  case 1:
    to_send += "1";
    break;

  case 2:
    to_send += "2";
    break;

  case 3:
    to_send += "3";
    break;

  default:
    to_send += "0";
    break;
  }

  if (command == 'S')
  {
    SendCommandToFireBase("Solve");
    Serial.println("Solve");
    executed = true;
  }

  else
  {
    SendCommandToFireBase(to_send);
    Serial.println(to_send);
    executed = true;
  }

  delay(1000);
  SendCommandToFireBase("None");

  return executed; // returns true if done moving
}

void loop()
{
  String commands = "None";

  bool is_wifi_enabled = true;

  while (!communication_handler.read_command())
  {
    if (HIGH == digitalRead(WIFIPIN) && is_wifi_enabled)
    {
      WiFi.disconnect();
      is_wifi_enabled = false;
      Serial.println("WiFi disabled");
    }

    else if (is_wifi_enabled && Firebase.RTDB.getString(&fbdo, "/solver"))
    {
      if (fbdo.dataType() == "string")
      {
        commands = fbdo.stringData();
        if (commands != "None")
        {
          Serial.println(commands);
          Serial2.println(commands);

          // TODO delay

          if (Firebase.ready() && signupOK)
          {
            String path = "/solver";
            if (!Firebase.RTDB.setString(&fbdo, path, "None"))
            {
              Serial.println("FAILED");
              Serial.println("REASON: " + fbdo.errorReason());
            }
          }
        }
      }

      commands = "None";
    }

    else if (is_wifi_enabled && Firebase.RTDB.getString(&fbdo, "/scramble"))
    {
      if (fbdo.dataType() == "string")
      {
        commands = fbdo.stringData();
        if (commands != "None")
        {
          Serial.println(commands);
          Serial2.println(commands);

          // TODO delay

          if (Firebase.ready() && signupOK)
          {
            String path = "/scramble";
            if (!Firebase.RTDB.setString(&fbdo, path, "None"))
            {
              Serial.println("FAILED");
              Serial.println("REASON: " + fbdo.errorReason());
            }
          }
        }
      }

      commands = "None";
    }

    else if (is_wifi_enabled && Firebase.RTDB.getString(&fbdo, "/hint"))
    {
      if (fbdo.dataType() == "string")
      {
        commands = fbdo.stringData();
        if (commands != "None")
        {
          Serial.println(commands);
          Serial2.println(commands);

          // TODO delay

          if (Firebase.ready() && signupOK)
          {
            String path = "/hint";
            if (!Firebase.RTDB.setString(&fbdo, path, "None"))
            {
              Serial.println("FAILED");
              Serial.println("REASON: " + fbdo.errorReason());
            }
          }
        }
      }

      commands = "None";
    }
  }

  Serial.println("we passed");

  if (!is_wifi_enabled)
  {
    WiFi.reconnect();
    Serial.print("reconnecting to WiFi");
    while (WiFi.status() != WL_CONNECTED)
    {
      Serial.print(".");
      delay(100);
    }
    Serial.println("WiFi reconnected");
  }

  while (!execute_comand(communication_handler.get_cmd(), communication_handler.get_indicator()))
    ;
}
