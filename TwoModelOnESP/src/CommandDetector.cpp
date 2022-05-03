#include <Arduino.h>
#include "I2SSampler.h"
#include "AudioProcessorWakeWord.h"
#include "AudioProcessorCommand.h"
#include "NeuralNetworkWakeWord.h"
#include "NeuralNetworkCommand.h"
#include "RingBuffer.h"
#include "CommandDetector.h"
#include "CommandProcessor.h"

#define WINDOW_SIZE 320
#define STEP_SIZE 160
#define POOLING_SIZE 6
#define AUDIO_LENGTH 16000
#define DETECTION_THRESHOLD -3
#define WAIT_PERIOD 500

CommandDetector::CommandDetector(I2SSampler *sample_provider, CommandProcessor *command_procesor)
{
    m_command_processor = command_procesor;

    // save the sample provider for use later
    m_sample_provider = sample_provider;

    /* Create for WakeWord */
    m_nn_wake_word = new NeuralNetworkWakeWord();
    Serial.println("Created Neural Network Wake Word");
    // create our audio processor
    m_audio_processor_wake_word = new AudioProcessorWakeWord(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
    Serial.println("Created audio processor");
    m_last_detection = 0;

    /* Create for Command */
    // Create our neural network
    m_nn_command = new NeuralNetworkCommand();
    Serial.println("Created Neural Network Command");
    // create our audio processor
    m_audio_processor_command = new AudioProcessorCommand(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
    // clear down the window
    for (int i = 0; i < COMMAND_WINDOW; i++)
    {
        for (int j = 0; j < NUMBER_COMMANDS; j++)
        {
            m_scores[i][j] = 0;
        }
    }
    m_scores_index = 0;

    Serial.println("Created audio processor coomand");

    isWakeWord = true;
}

CommandDetector::~CommandDetector()
{
    delete m_nn_wake_word;
    m_nn_wake_word = NULL;
    delete m_audio_processor_wake_word;
    m_audio_processor_wake_word = NULL;
    delete m_nn_command;
    m_nn_command = NULL;
    delete m_audio_processor_command;
    m_audio_processor_command = NULL;
    uint32_t free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram after DetectWakeWord cleanup %d\n", free_ram);
}

void CommandDetector::run()
{
    if (isWakeWord)
    {
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
            delay(200); // move to the next sector, so give it some time

            digitalWrite(GPIO_NUM_2, LOW);
        }
        // nothing detected stay in the current state
    }

    else
    {
        // time how long this takes for stats
        long start = millis();
        // get access to the samples that have been read in
        RingBufferAccessor *reader = m_sample_provider->getRingBufferReader();
        // rewind by 1 second
        reader->rewind(16000);
        // get hold of the input buffer for the neural network so we can feed it data
        float *input_buffer = m_nn_command->getInputBufferCommand();
        // process the samples to get the spectrogram
        m_audio_processor_command->get_spectrogramCommand(reader, input_buffer);
        // finished with the sample reader
        delete reader;
        // get the prediction for the spectrogram
        float output = m_nn_command->predictCommand();
        long end = millis();

        // use quite a high threshold to prevent false positives
        if (output > 0.95 && start - m_last_detection > WAIT_PERIOD)
        {
            // detected the wake word in several runs, move to the next state
            m_last_detection = start;
            Serial.printf("P(%.2f): Detect the word marvin\n", output);

            digitalWrite(GPIO_NUM_2, HIGH);

            isWakeWord = true;
            delay(200); // move to the next sector, so give it some time

            digitalWrite(GPIO_NUM_2, LOW);
        }
        // nothing detected stay in the current state
    }
}
