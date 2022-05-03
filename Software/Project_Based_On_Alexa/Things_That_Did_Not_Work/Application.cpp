#include <Arduino.h>
#include "Application.h"
#include "state_machine/DetectWakeWordState.h"
#include "state_machine/RecogniseCommandState.h"
#include "IndicatorLight.h"
#include "IntentProcessor.h"
#include "I2SMicSampler.h"
#include "config.h"


Application::Application(I2SSampler *sample_provider, IntentProcessor *intent_processor, IndicatorLight *indicator_light)
{
    // detect wake word state - waits for the wake word to be detected
    m_detect_wake_word_state = new DetectWakeWordState(sample_provider);
    // command recongiser - streams audio to the server for recognition
    
    m_sample_provider = sample_provider;
    intent_processor_provider = intent_processor;
    indicator_light_provider = indicator_light;
    //m_recognise_command_state = new RecogniseCommandState(sample_provider, indicator_light, intent_processor);
    
    // start off in the detecting wakeword state
    m_current_state = m_detect_wake_word_state;
    m_current_state->enterState();
}

// process the next batch of samples
void Application::run()
{
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
    
    bool state_done = m_current_state->run();
    if (state_done)
    {
        m_current_state->exitState();
        // switch to the next state - very simple state machine so we just go to the other state...
        
        I2SSampler *i2s_sampler = new I2SMicSampler(i2s_mic_pins, false);
        
        m_sample_provider = i2s_sampler;

        if (m_current_state == m_detect_wake_word_state)
        {
            m_recognise_command_state = new RecogniseCommandState(m_sample_provider, indicator_light_provider, intent_processor_provider);
            m_current_state = m_recognise_command_state;
        }
        else
        {
            m_detect_wake_word_state = new DetectWakeWordState(m_sample_provider);
            m_current_state = m_detect_wake_word_state;
        }
        m_current_state->enterState();
    }
    vTaskDelay(10);
}
