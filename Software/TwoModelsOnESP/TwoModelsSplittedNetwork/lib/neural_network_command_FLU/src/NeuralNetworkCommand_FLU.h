#ifndef __NeuralNetwork_FLU__
#define __NeuralNetwork_FLU__

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
} NNResult_FLU;

class NeuralNetworkCommand_FLU
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
    NeuralNetworkCommand_FLU();
    ~NeuralNetworkCommand_FLU();
    float *getInputBufferCommand_FLU();
    float *getOutputBufferCommand_FLU();
    NNResult_FLU predictCommand_FLU();
};

#endif