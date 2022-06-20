#include <inttypes.h>

class HammingWindowCommand_GL
{
private:
    float *m_coefficients;
    int m_window_size;

public:
    HammingWindowCommand_GL(int window_size);
    ~HammingWindowCommand_GL();
    void applyWindowCommand_GL(float *input);
};