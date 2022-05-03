#include <Arduino.h>
#include "I2SSampler.h"
#include "AudioProcessorWakeWord.h"
#include "NeuralNetworkWakeWord.h"
#include "RingBuffer.h"
#include "DetectWakeWordState.h"

#define WINDOW_SIZE 320
#define STEP_SIZE 160
#define POOLING_SIZE 6
#define AUDIO_LENGTH 16000
#define WAIT_PERIOD 1000

DetectWakeWordState::DetectWakeWordState(I2SSampler *sample_provider)
{
    // save the sample provider for use later
    m_sample_provider = sample_provider;
}
void DetectWakeWordState::enterState()
{
    // Create our neural network
    m_nn = new NeuralNetworkWakeWord();
    Serial.println("Created Neural Network WakeWord");
    // create our audio processor
    m_audio_processor = new AudioProcessorWakeWord(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
    Serial.println("Created audio processor");

    m_last_audio_position = -1;
    Serial.println("Ready for action");
}
bool DetectWakeWordState::run()
{
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

        float *input_buffer = m_nn->getInputBufferWakeWord();
        // process the samples to get the spectrogram
        m_audio_processor->get_spectrogramWakeWord(reader, input_buffer);
        // finished with the sample reader
        delete reader;
        // get the prediction for the spectrogram
        float output = m_nn->predictWakeWord();
        long end = millis();

        // use quite a high threshold to prevent false positives
        if (output > 0.95 && start - m_last_detection > WAIT_PERIOD)
        {
            m_last_detection = start;

            // detected the wake word in several runs, move to the next state
            Serial.printf("P(%.2f): Detected wake word 'Go'...\n", output);
            return true;
        }
        // nothing detected stay in the current state
        return false;
    }

    // nothing detected stay in the current state
    return false;
}
void DetectWakeWordState::exitState()
{
    // Create our neural network
    delete m_nn;
    m_nn = NULL;
    delete m_audio_processor;
    m_audio_processor = NULL;
    uint32_t free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram after DetectWakeWord cleanup %d\n", free_ram);
}