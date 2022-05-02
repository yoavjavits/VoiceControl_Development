#ifndef _recognise_command_state_h_
#define _recognise_command_state_h_

#include "States.h"

class I2SSampler;

class WiFiClient;
class HTTPClient;
class IndicatorLight;
class IntentProcessor;

class NeuralNetworkCommand;
class AudioProcessorCommand;

#define NUMBER_COMMANDS 5
#define COMMAND_WINDOW 3

class RecogniseCommandState : public State
{
private:
    I2SSampler *m_sample_provider;
    int m_last_audio_position;

    NeuralNetworkCommand *m_nn;
    AudioProcessorCommand *m_audio_processor;

    IndicatorLight *m_indicator_light;
    IntentProcessor *m_intent_processor;

    int m_number_of_runs;
    float m_scores[COMMAND_WINDOW][NUMBER_COMMANDS];
    int m_scores_index;
    unsigned long m_last_detection;

public:
    RecogniseCommandState(I2SSampler *sample_provider, IndicatorLight *indicator_light, IntentProcessor *intent_processor);
    void enterState();
    bool run();
    void exitState();
};

#endif
