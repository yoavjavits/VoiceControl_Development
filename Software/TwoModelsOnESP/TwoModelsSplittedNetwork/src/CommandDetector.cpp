#include <Arduino.h>
#include "I2SSampler.h"
#include "AudioProcessorWakeWord.h"
#include "AudioProcessorCommand_FLU.h"
#include "AudioProcessorCommand_BRU.h"
#include "../lib/neural_network_command_FLU/src/NeuralNetworkCommand_FLU.h"
#include "../lib/neural_network_command_BRU/src/NeuralNetworkCommand_BRU.h"
#include "../lib/neural_network_wake_word/src/NeuralNetworkWakeWord.h"
#include "RingBuffer.h"
#include "CommandDetector.h"
#include "CommandProcessor.h"

#define WINDOW_SIZE 320
#define STEP_SIZE 160
#define POOLING_SIZE 6
#define AUDIO_LENGTH 16000
#define DETECTION_THRESHOLD -3
#define WAIT_PERIOD 1000

CommandDetector::CommandDetector(I2SSampler *sample_provider, CommandProcessor *command_procesor)
{
    m_command_processor = command_procesor;

    // save the sample provider for use later
    m_sample_provider = sample_provider;
    m_last_detection = 0;

    isWakeWord = true;
    first_time = true;

    for (int i = 0; i < COMMAND_WINDOW; i++)
    {
        for (int j = 0; j < NUMBER_COMMANDS; j++)
        {
            m_scores_FLU[i][j] = 0;
            m_scores_BRU[i][j] = 0;
        }
    }
    m_scores_index_FLU = 0;
    m_scores_index_BRU = 0;

    // Serial.println("Created audio processor coomand");
}

CommandDetector::~CommandDetector()
{
}

void CommandDetector::run()
{
    if (isWakeWord)
    {
        if (first_time)
        {
            /* Create for WakeWord */
            m_nn_wake_word = new NeuralNetworkWakeWord();
            Serial.println("Created Neural Network Wake Word");
            // create our audio processor
            m_audio_processor_wake_word = new AudioProcessorWakeWord(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
            first_time = false;
        }

        // time how long this takes for stats
        long start = millis();
        // get access to the samples that have been read in
        RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
        // rewind by 1 second
        reader->rewind(16000);
        // get hold of the input buffer for the neural network so we can feed it data
        float *input_buffer = m_nn_wake_word->getInputBufferWakeWord();
        // process the samples to get the spectrogram
        m_audio_processor_wake_word->get_spectrogramWakeWord(reader, input_buffer);
        // finished with the sample reader
        delete reader;
        // get the prediction for the spectrogram
        float output = m_nn_wake_word->predictWakeWord();
        long end = millis();

        // use quite a high threshold to prevent false positives
        if (output > 0.95 && start - m_last_detection > WAIT_PERIOD)
        {
            // detected the wake word in several runs, move to the next state
            m_last_detection = start;
            Serial.printf("P(%.2f): Detect the word go\n", output);

            digitalWrite(GPIO_NUM_2, HIGH);
            isWakeWord = false;
            first_time = true;
            delay(750); // move to the next sector, so give it some time
            digitalWrite(GPIO_NUM_2, LOW);

            delete m_nn_wake_word;
            m_nn_wake_word = NULL;
            delete m_audio_processor_wake_word;
            m_audio_processor_wake_word = NULL;
            Serial.println("destroyed Neural Network Wake Word");
        }
    }

    else
    {
        if (first_time)
        {
            m_nn_command_FLU = new NeuralNetworkCommand_FLU();
            m_nn_command_BRU = new NeuralNetworkCommand_BRU();

            Serial.println("Created Neural Network Command");
            // create our audio processor
            m_audio_processor_command_FLU = new AudioProcessorCommand_FLU(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
            m_audio_processor_command_BRU = new AudioProcessorCommand_BRU(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
            first_time = false;
        }

        // time how long this takes for stats
        long start = millis();
        // get access to the samples that have been read in
        RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
        // rewind by 1 second
        reader->rewind(16000);
        // get hold of the input buffer for the neural network so we can feed it data
        float *input_buffer_FLU = m_nn_command_FLU->getInputBufferCommand_FLU();
        float *input_buffer_BRU = m_nn_command_BRU->getInputBufferCommand_BRU();
        // process the samples to get the spectrogram
        bool is_valid_FLU = m_audio_processor_command_FLU->get_spectrogramCommand_FLU(reader, input_buffer_FLU);
        bool is_valid_BRU = m_audio_processor_command_BRU->get_spectrogramCommand_BRU(reader, input_buffer_BRU);
        // finished with the sample reader
        delete reader;
        // get the prediction for the spectrogram
        m_nn_command_FLU->predictCommand_FLU();
        m_nn_command_BRU->predictCommand_BRU();

        ///*** Analysis of FLU ***///

        // keep track of the previous 5 scores - about 0.5 seconds given current processing speed
        for (int i = 0; i < NUMBER_COMMANDS; i++)
        {
            float prediction_FLU = std::max(m_nn_command_FLU->getOutputBufferCommand_FLU()[i], 1e-6f);
            m_scores_FLU[m_scores_index_FLU][i] = log(is_valid_FLU ? prediction_FLU : 1e-6);
        }
        m_scores_index_FLU = (m_scores_index_FLU + 1) % COMMAND_WINDOW;
        // get the best score
        float scores_FLU[NUMBER_COMMANDS] = {0, 0, 0, 0};
        for (int i = 0; i < COMMAND_WINDOW; i++)
        {
            for (int j = 0; j < NUMBER_COMMANDS; j++)
            {
                scores_FLU[j] += m_scores_FLU[i][j];
            }
        }
        // get the best score
        float best_score_FLU = scores_FLU[0];
        int best_index_FLU = 0;
        for (int i = 1; i < NUMBER_COMMANDS; i++)
        {
            if (scores_FLU[i] > best_score_FLU)
            {
                best_index_FLU = i;
                best_score_FLU = scores_FLU[i];
            }
        }

        ///*** Analysis of BRU ***///

        // keep track of the previous 5 scores - about 0.5 seconds given current processing speed
        for (int i = 0; i < NUMBER_COMMANDS; i++)
        {
            float prediction_BRU = std::max(m_nn_command_BRU->getOutputBufferCommand_BRU()[i], 1e-6f);
            m_scores_BRU[m_scores_index_BRU][i] = log(is_valid_BRU ? prediction_BRU : 1e-6);
        }
        m_scores_index_BRU = (m_scores_index_BRU + 1) % COMMAND_WINDOW;
        // get the best score
        float scores_BRU[NUMBER_COMMANDS] = {0, 0, 0, 0};
        for (int i = 0; i < COMMAND_WINDOW; i++)
        {
            for (int j = 0; j < NUMBER_COMMANDS; j++)
            {
                scores_BRU[j] += m_scores_BRU[i][j];
            }
        }
        // get the best score
        float best_score_BRU = scores_BRU[0];
        int best_index_BRU = 0;
        for (int i = 1; i < NUMBER_COMMANDS; i++)
        {
            if (scores_BRU[i] > best_score_BRU)
            {
                best_index_BRU = i;
                best_score_BRU = scores_BRU[i];
            }
        }

        long end = millis();

        bool check = best_score_FLU > best_score_BRU;
        float best_score = check ? best_score_FLU : best_index_BRU;
        int best_index = check ? best_index_FLU : best_index_BRU;

        // sanity check best score and check the cool down period
        if (best_score > DETECTION_THRESHOLD && best_index != NUMBER_COMMANDS - 1 && start - m_last_detection > WAIT_PERIOD)
        {
            m_last_detection = start;

            if (check)
            {
                switch (best_index)
                {
                case 0:
                    Serial.printf("P(%.2f): Detect the word %s\n", best_score, "forward");
                    break;

                case 1:
                    Serial.printf("P(%.2f): Detect the word %s\n", best_score, "up'");
                    break;

                case 2:
                    Serial.printf("P(%.2f): Detect the word %s\n", best_score, "left");
                    break;

                default:
                    break;
                }
            }
            else
            {
                switch (best_index)
                {
                case 0:
                    Serial.printf("P(%.2f): Detect the word %s\n", best_score, "backward");
                    break;

                case 1:
                    Serial.printf("P(%.2f): Detect the word %s\n", best_score, "right");
                    break;

                case 2:
                    Serial.printf("P(%.2f): Detect the word %s\n", best_score, "down");
                    break;

                default:
                    break;
                }
            }

            digitalWrite(GPIO_NUM_2, HIGH);
            isWakeWord = true;
            first_time = true;
            delay(500);
            digitalWrite(GPIO_NUM_2, LOW);

            delete m_nn_command_FLU;
            m_nn_command_FLU = NULL;
            delete m_audio_processor_command_FLU;
            m_audio_processor_command_FLU = NULL;

            delete m_nn_command_BRU;
            m_nn_command_BRU = NULL;
            delete m_audio_processor_command_BRU;
            m_audio_processor_command_BRU = NULL;

            Serial.println("destroyed Neural Network Command");

            m_command_processor->queueCommand(best_index, best_score); // to add check boolean
        }
    }
}
