#include <Arduino.h>
#include "I2SSampler.h"
#include "AudioProcessor.h"
#include "NeuralNetwork.h"
#include "RingBuffer.h"
#include "WakeWordDetector.h"

#define WINDOW_SIZE 320
#define STEP_SIZE 160
#define POOLING_SIZE 6
#define AUDIO_LENGTH 16000
#define WAIT_PERIOD 1000

DetectWakeWord::DetectWakeWord(I2SSampler *sample_provider)
{
    // save the sample provider for use later
    m_sample_provider = sample_provider;
    // some stats on performance
    m_last_detection = 0;

     // Create our neural network
    m_nn = new NeuralNetwork();
    Serial.println("Created Neural Network");
    // create our audio processor
    m_audio_processor = new AudioProcessor(AUDIO_LENGTH, WINDOW_SIZE, STEP_SIZE, POOLING_SIZE);
    Serial.println("Created audio processor");
}

void DetectWakeWord::run()
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
    // compute the stats
    m_average_detect_time = (end - start) * 0.1 + m_average_detect_time * 0.9;

    // log out some timing info
    /*if (m_number_of_runs == 100)
    {
        m_number_of_runs = 0;
        Serial.printf("Average detection time %.fms\n", m_average_detect_time);
    }*/

    // use quite a high threshold to prevent false positives
    if (output > 0.95 && start - m_last_detection > WAIT_PERIOD)
    {
        m_last_detection = start;
    
        Serial.printf("P(%.2f): Detected wake word 'Go'...\n", output);

        digitalWrite(GPIO_NUM_2, HIGH);
        delay(100);
        //TODO: change this to another way because it makes the process slower!!!
        digitalWrite(GPIO_NUM_2, LOW);
        //TODO: do what we want do to do.
        
    }
}


/*
if (output > 0.95 && start - m_last_detection > WAIT_PERIOD)
    {
        m_last_detection = start;
        m_number_of_detections++;

        if (m_number_of_detections > 1)
        {
            m_number_of_detections = 0;
            Serial.printf("P(%.2f): Detected wake word 'Go'...\n", output);

            digitalWrite(GPIO_NUM_2, HIGH);   // turn the LED on (HIGH is the voltage level)
            delay(200);                       // wait for a second
            digitalWrite(GPIO_NUM_2, LOW);    // turn the LED off by making the voltage LOW
            //TODO: do what we want do to do.
        }
    }
*/

DetectWakeWord::~DetectWakeWord()
{
    // Create our neural network
    delete m_nn;
    m_nn = NULL;
    delete m_audio_processor;
    m_audio_processor = NULL;
    uint32_t free_ram = esp_get_free_heap_size();
    Serial.printf("Free ram after DetectWakeWord cleanup %d\n", free_ram);
}