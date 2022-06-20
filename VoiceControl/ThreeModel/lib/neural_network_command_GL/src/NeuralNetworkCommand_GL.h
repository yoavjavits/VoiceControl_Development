#ifndef __NeuralNetwork_GL__
#define __NeuralNetwork_GL__

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
} NNResult_GL;

class NeuralNetworkCommand_GL
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
    NeuralNetworkCommand_GL();
    ~NeuralNetworkCommand_GL();
    float *getInputBufferCommand_GL();
    float *getOutputBufferCommand_GL();
    NNResult_GL predictCommand_GL();
};

#endif