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
}
void RecogniseCommandState::enterState()
{
    // Create our neural network
    m_nn = new NeuralNetworkCommand();
    Serial.println("Created Neural Network Command");
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

    // indicate that we are now recording audio
    m_indicator_light->setState(ON);

    m_last_audio_position = -1;

    // uint32_t free_ram = esp_get_free_heap_size();
    // Serial.printf("Free ram before connection %d\n", free_ram);

    Serial.println("Ready for action");

    // free_ram = esp_get_free_heap_size();
    // Serial.printf("Free ram after connection %d\n", free_ram);
}
bool RecogniseCommandState::run()
{
    //Serial.println("run command");
    // time how long this takes for stats
    long start = millis();

    if (m_last_audio_position == -1)
    {
        // set to 1 seconds in the past the allow for the really slow connection time
        m_last_audio_position = m_sample_provider->getCurrentWritePosition() - 16000;
    }
    // how many samples have been captured since we last ran
    int audio_position = m_sample_provider->getCurrentWritePosition();
    // work out how many samples there are taking into account that we can wrap around
    int sample_count = (audio_position - m_last_audio_position + m_sample_provider->getRingBufferSize()) % m_sample_provider->getRingBufferSize();
    // Serial.printf("Last sample position %d, current position %d, number samples %d\n", m_last_audio_position, audio_position, sample_count);

    if (sample_count > 0)
    {
        RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
        reader->setIndex(m_last_audio_position);

        /*
        // get access to the samples that have been read in
        RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
        // rewind by 1 second
        reader->rewind(16000);
        // get hold of the input buffer for the neural network so we can feed it data
        */

        float *input_buffer = m_nn->getInputBufferCommand();
        // process the samples to get the spectrogram
        bool is_valid = m_audio_processor->get_spectrogramCommand(reader, input_buffer);
        // finished with the sample reader
        m_last_audio_position = reader->getIndex();
        delete reader;
        // get the prediction for the spectrogram

        NNResult output = m_nn->predictCommand();
        long end = millis();

        // use quite a high threshold to prevent false positives
        if (output.score > 0.95 && start - m_last_detection > WAIT_PERIOD)
        {
            int index = output.index;
            m_last_detection = start;

            // detected the wake word in several runs, move to the next state
            Serial.printf("P(%.2f): Detected %d...\n", output.score, output.index);
            return true;
        }

        /*
        m_nn->predictCommand();
        // keep track of the previous 5 scores - about 0.5 seconds given current processing speed
        for (int i = 0; i < NUMBER_COMMANDS; i++)
        {
            float prediction = std::max(m_nn->getOutputBufferCommand()[i], 1e-6f);
            m_scores[m_scores_index][i] = log(is_valid ? prediction : 1e-6);
        }
        m_scores_index = (m_scores_index + 1) % COMMAND_WINDOW;
        // get the best score
        float scores[NUMBER_COMMANDS] = {0, 0, 0, 0, 0};
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
            Serial.println("detected command");

            //m_command_processor->queueCommand(best_index, best_score);
            return true;
        }*/

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
    }
    
    // nothing detected stay in the current state
    return false;
}

void RecogniseCommandState::exitState()
{
    delete m_nn;
    m_nn = NULL;
    delete m_audio_processor;
    m_audio_processor = NULL;
    // uint32_t free_ram = esp_get_free_heap_size();
    // Serial.printf("Free ram after DetectCommand cleanup %d\n", free_ram);
}