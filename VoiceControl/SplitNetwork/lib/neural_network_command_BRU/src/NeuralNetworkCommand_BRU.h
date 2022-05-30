#ifndef __NeuralNetwork_BRU__
#define __NeuralNetwork_BRU__

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

typedef struct
{
    float score;
    int index;
} NNResult_BRU;

class NeuralNetworkCommand_BRU
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
    NeuralNetworkCommand_BRU();
    ~NeuralNetworkCommand_BRU();
    float *getInputBufferCommand_BRU();
    float *getOutputBufferCommand_BRU();
    NNResult_BRU predictCommand_BRU();
};

#endif