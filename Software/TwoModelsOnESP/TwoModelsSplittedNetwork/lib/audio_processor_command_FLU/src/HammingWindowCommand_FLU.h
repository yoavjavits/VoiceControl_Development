#include <inttypes.h>

class HammingWindowCommand_FLU
{
private:
    float *m_coefficients;
    int m_window_size;

public:
    HammingWindowCommand_FLU(int window_size);
    ~HammingWindowCommand_FLU();
    void applyWindowCommand_FLU(float *input);
};