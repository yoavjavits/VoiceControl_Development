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
#include "AudioProcessorCommand.h"
#include "NeuralNetworkCommand.h"
#include "RingBuffer.h"

#define WINDOW_SIZE 320
#define STEP_SIZE 160
#define POOLING_SIZE 6
#define AUDIO_LENGTH 16000
#define DETECTION_THRESHOLD -3
#define WAIT_PERIOD 1000

RecogniseCommandState::RecogniseCommandState(I2SSampler *sample_provider, IndicatorLight *indicator_light, IntentProcessor *intent_processor)
{
    // save the sample provider for use later
    m_sample_provider = sample_provider;
    m_indicator_light = indicator_light;
    m_intent_processor = intent_processor;

    m_last_detection = 0;
    // m_command_processor = command_procesor;

    // Create our neural network
    m_nn = new NeuralNetwork();
    Serial.println("Created Neural Network");
    // create our audio processor
    m_audio_processor = new AudioProcessorCommand(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
    // clear down the window
    for (int i = 0; i < COMMAND_WINDOW; i++)
    {
        for (int j = 0; j < NUMBER_COMMANDS; j++)
        {
            m_scores[i][j] = 0;
        }
    }
    m_scores_index = 0;

    Serial.println("Created audio processor");
}
void RecogniseCommandState::enterState()
{
    // indicate that we are now recording audio
    m_indicator_light->setState(ON);

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
    bool is_valid = m_audio_processor->get_spectrogramCommand(reader, input_buffer);
    // finished with the sample reader
    delete reader;
    // get the prediction for the spectrogram
    m_nn->predict();
    // keep track of the previous 5 scores - about 0.5 seconds given current processing speed
    for (int i = 0; i < NUMBER_COMMANDS; i++)
    {
        float prediction = std::max(m_nn->getOutputBuffer()[i], 1e-6f);
        m_scores[m_scores_index][i] = log(is_valid ? prediction : 1e-6);
    }
    m_scores_index = (m_scores_index + 1) % COMMAND_WINDOW;
    // get the best score
    float scores[NUMBER_COMMANDS] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < COMMAND_WINDOW; i++)
    {
        for (int j = 0; j < NUMBER_COMMANDS; j++)
        {
            scores[j] += m_scores[i][j];
        }
    }
    // get the best score
    float best_score = scores[0];
    int best_index = 0;
    for (int i = 1; i < NUMBER_COMMANDS; i++)
    {
        if (scores[i] > best_score)
        {
            best_index = i;
            best_score = scores[i];
        }
    }

    long end = millis();
    // sanity check best score and check the cool down period
    if (best_score > DETECTION_THRESHOLD && best_index != NUMBER_COMMANDS - 1 && start - m_last_detection > WAIT_PERIOD)
    {
        m_last_detection = start;
        
        //m_command_processor->queueCommand(best_index, best_score);
        return true;
    }

    /*
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
    return true; */
    
    // still work to do, stay in this state
    return false;
}

void RecogniseCommandState::exitState()
{
    // clean up the speech recognizer client as it takes up a lot of RAM
    uint32_t free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram after request %d\n", free_ram);
}