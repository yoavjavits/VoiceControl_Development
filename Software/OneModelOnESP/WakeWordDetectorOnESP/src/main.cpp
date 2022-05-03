#include <Arduino.h>
//#include <WiFi.h>
#include <driver/i2s.h>
#include <esp_task_wdt.h>
#include "I2SMicSampler.h"
//#include "ADCSampler.h"
//#include "I2SOutput.h"
#include "config.h"
#include "WakeWordDetector.h"
//#include "Application.h"
//#include "SPIFFS.h"
//#include "IntentProcessor.h"
//#include "Speaker.h"
//#include "IndicatorLight.h"

// i2s config for reading from both channels of I2S
i2s_config_t i2sMemsConfigBothChannels = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_MIC_CHANNEL,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0};

// i2s microphone pins
i2s_pin_config_t i2s_mic_pins = {
    .bck_io_num = I2S_MIC_SERIAL_CLOCK,
    .ws_io_num = I2S_MIC_LEFT_RIGHT_CLOCK,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_MIC_SERIAL_DATA};

// This task does all the heavy lifting for our application
void applicationTask(void *param)
{
  DetectWakeWord *detectWakeWord = static_cast<DetectWakeWord *>(param);

  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    // wait for some audio samples to arrive
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      detectWakeWord->run();
    }
  }
}

void setup()
{
  Serial.begin(9600);
  delay(1000);
  Serial.println("Starting up");

  Serial.printf("Total heap: %d\n", ESP.getHeapSize());
  Serial.printf("Free heap: %d\n", ESP.getFreeHeap());

  pinMode(GPIO_NUM_2, OUTPUT);
  
  // make sure we don't get killed for our long running tasks
  esp_task_wdt_init(10, false);
  I2SSampler *i2s_sampler = new I2SMicSampler(i2s_mic_pins, false);

  DetectWakeWord *detectWakeWord = new DetectWakeWord(i2s_sampler);

  // set up the i2s sample writer task
  TaskHandle_t applicationTaskHandle;
  xTaskCreatePinnedToCore(applicationTask, "Detect Wake Word", 8192, detectWakeWord, 1, &applicationTaskHandle, 0);

  // start sampling from i2s device - use I2S_NUM_0 as that's the one that supports the internal ADC
  i2s_sampler->start(I2S_NUM_0, i2sMemsConfigBothChannels, applicationTaskHandle);
}

void loop()
{
  vTaskDelay(pdMS_TO_TICKS(1000));
  //vTaskDelay(1000);
}