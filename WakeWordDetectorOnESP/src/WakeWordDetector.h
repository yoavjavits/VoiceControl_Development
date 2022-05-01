#ifndef _detect_wake_word_state_h_
#define _detect_wake_word_state_h_

class I2SSampler;
class NeuralNetwork;
class AudioProcessor;

class DetectWakeWord
{
private:
    I2SSampler *m_sample_provider;
    NeuralNetwork *m_nn;
    AudioProcessor *m_audio_processor;
    float m_average_detect_time;
    int m_number_of_detections;
    int m_number_of_runs;
    unsigned long m_last_detection;

public:
    DetectWakeWord(I2SSampler *sample_provider);
    ~DetectWakeWord();
    void run();
};

#endif
