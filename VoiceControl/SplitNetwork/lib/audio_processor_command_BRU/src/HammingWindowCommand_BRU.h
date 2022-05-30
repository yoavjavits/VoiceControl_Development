#include <inttypes.h>

class HammingWindowCommand_BRU
{
private:
    float *m_coefficients;
    int m_window_size;

public:
    HammingWindowCommand_BRU(int window_size);
    ~HammingWindowCommand_BRU();
    void applyWindowCommand_BRU(float *input);
};