#include <inttypes.h>

class HammingWindowWakeWord
{
private:
    float *m_coefficients;
    int m_window_size;

public:
    HammingWindowWakeWord(int window_size);
    ~HammingWindowWakeWord();
    void applyWindowWakeWord(float *input);
};