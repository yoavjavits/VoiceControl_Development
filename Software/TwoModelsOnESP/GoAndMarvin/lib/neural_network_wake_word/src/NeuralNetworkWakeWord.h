#ifndef __NeuralNetworkWakeWord__
#define __NeuralNetworkWakeWord__

#include <stdint.h>

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class NeuralNetworkWakeWord
{
private:
    tflite::MicroMutableOpResolver<10> *m_resolver;
    tflite::ErrorReporter *m_error_reporter;
    const tflite::Model *m_model;
    tflite::MicroInterpreter *m_interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *m_tensor_arena;

public:
    NeuralNetworkWakeWord();
    ~NeuralNetworkWakeWord();
    float *getInputBufferWakeWord();
    float predictWakeWord();
};

#endif