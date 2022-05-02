#ifndef _recognise_command_state_h_
#define _recognise_command_state_h_

#include "States.h"

class I2SSampler;
class WiFiClient;
class HTTPClient;
class IndicatorLight;
class IntentProcessor;

class NeuralNetwork;
class AudioProcessor;

class RecogniseCommandState : public State
{
private:
    I2SSampler *m_sample_provider;
    unsigned long m_start_time;
    unsigned long m_elapsed_time;
    int m_last_audio_position;

    IndicatorLight *m_indicator_light;
    IntentProcessor *m_intent_processor;

    NeuralNetwork *m_nn;
    AudioProcessor *m_audio_processor;
    unsigned long m_last_detection;

public:
    RecogniseCommandState(I2SSampler *sample_provider, IndicatorLight *indicator_light, IntentProcessor *intent_processor);
    void enterState();
    bool run();
    void exitState();
};

#endif
