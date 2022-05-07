#ifndef _detect_wake_word_state_h_
#define _detect_wake_word_state_h_

class I2SSampler;

class NeuralNetworkWakeWord;
class AudioProcessorWakeWord;

class NeuralNetworkCommand;
class AudioProcessorCommand;

class CommandProcessor;

#define NUMBER_COMMANDS 7
#define COMMAND_WINDOW 3

class CommandDetector
{
private:
    CommandProcessor *m_command_processor;
    I2SSampler *m_sample_provider;

    NeuralNetworkWakeWord *m_nn_wake_word;
    AudioProcessorWakeWord *m_audio_processor_wake_word;
    unsigned long m_last_detection;

    NeuralNetworkCommand *m_nn_command;
    AudioProcessorCommand *m_audio_processor_command;
    float m_scores[COMMAND_WINDOW][NUMBER_COMMANDS];
    int m_scores_index;

    bool isWakeWord;
    bool first_time;

public:
    CommandDetector(I2SSampler *sample_provider, CommandProcessor *command_processor);
    ~CommandDetector();
    void run();
};

#endif
