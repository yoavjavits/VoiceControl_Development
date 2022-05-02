#include <Arduino.h>
#include <ArduinoJson.h>
#include "I2SSampler.h"
#include "RingBuffer.h"
#include "RecogniseCommandState.h"
#include "IndicatorLight.h"
#include "IntentProcessor.h"
#include "../config.h"
#include <string.h>

#include "I2SSampler.h"
#include "AudioProcessor.h"
#include "NeuralNetwork.h"
#include "RingBuffer.h"

#define WINDOW_SIZE 320
#define STEP_SIZE 160
#define POOLING_SIZE 6
#define AUDIO_LENGTH 16000

RecogniseCommandState::RecogniseCommandState(I2SSampler *sample_provider, IndicatorLight *indicator_light, IntentProcessor *intent_processor)
{
    // save the sample provider for use later
    m_sample_provider = sample_provider;
    m_indicator_light = indicator_light;
    m_intent_processor = intent_processor;
}
void RecogniseCommandState::enterState()
{
    // indicate that we are now recording audio
    m_indicator_light->setState(ON);

    // stash the start time - we will limit ourselves to 5 seconds of data
    m_start_time = millis();
    m_elapsed_time = 0;
    m_last_audio_position = -1;

    uint32_t free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram before connection %d\n", free_ram);

    Serial.println("Ready for action");

    free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram after connection %d\n", free_ram);
}
bool RecogniseCommandState::run()
{
    // time how long this takes for stats
    long start = millis();
    // get access to the samples that have been read in
    RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
    // rewind by 1 second
    reader->rewind(16000);
    // get hold of the input buffer for the neural network so we can feed it data
    float *input_buffer = m_nn->getInputBuffer();
    // process the samples to get the spectrogram
    m_audio_processor->get_spectrogram(reader, input_buffer);
    // finished with the sample reader
    delete reader;
    // get the prediction for the spectrogram
    float output = m_nn->predict();
    long end = millis();


    
        // has 3 seconds passed?
        unsigned long current_time = millis();
        m_elapsed_time += current_time - m_start_time;
        m_start_time = current_time;
        if (m_elapsed_time > 3000)
        {
            // indicate that we are now trying to understand the command
            m_indicator_light->setState(PULSING);

            // all done, move to next state
            Serial.println("3 seconds has elapsed - finishing recognition request");
            // final new line to finish off the request
            Intent intent = m_speech_recogniser->getResults();
            IntentResult intentResult = m_intent_processor->processIntent(intent);
            switch (intentResult)
            {
            case SUCCESS:
                m_speaker->playOK();
                break;
            case FAILED:
                m_speaker->playCantDo();
                break;
            case SILENT_SUCCESS:
                // nothing to do
                break;
            }
            // indicate that we are done
            m_indicator_light->setState(OFF);
            return true;
        }
    }
    // still work to do, stay in this state
    return false;
}

void RecogniseCommandState::exitState()
{
    // clean up the speech recognizer client as it takes up a lot of RAM
    delete m_speech_recogniser;
    m_speech_recogniser = NULL;
    uint32_t free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram after request %d\n", free_ram);
}