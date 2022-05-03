#include <Arduino.h>
#include "I2SSampler.h"
#include "AudioProcessorWakeWord.h"
#include "AudioProcessorCommand.h"
#include "../lib/neural_network_command/src/NeuralNetworkCommand.h"
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
            m_scores[i][j] = 0;
        }
    }
    m_scores_index = 0;

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
            m_nn_command = new NeuralNetworkCommand();
            Serial.println("Created Neural Network Command");
            // create our audio processor
            m_audio_processor_command = new AudioProcessorCommand(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
            first_time = false;
        }

        // time how long this takes for stats
        long start = millis();
        // get access to the samples that have been read in
        RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
        // rewind by 1 second
        reader->rewind(16000);
        // get hold of the input buffer for the neural network so we can feed it data
        float *input_buffer = m_nn_command->getInputBufferCommand();
        // process the samples to get the spectrogram
        bool is_valid = m_audio_processor_command->get_spectrogramCommand(reader, input_buffer);
        // finished with the sample reader
        delete reader;
        // get the prediction for the spectrogram
        m_nn_command->predictCommand();
        // keep track of the previous 5 scores - about 0.5 seconds given current processing speed
        for (int i = 0; i < NUMBER_COMMANDS; i++)
        {
            float prediction = std::max(m_nn_command->getOutputBufferCommand()[i], 1e-6f);
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
            digitalWrite(GPIO_NUM_2, HIGH);
            isWakeWord = true;
            first_time = true;
            delay(750);
            digitalWrite(GPIO_NUM_2, LOW);

            delete m_nn_command;
            m_nn_command = NULL;
            delete m_audio_processor_command;
            m_audio_processor_command = NULL;
            Serial.println("destroyed Neural Network Command");

            m_command_processor->queueCommand(best_index, best_score);
        }
    }
}
