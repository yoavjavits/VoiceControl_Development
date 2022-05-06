#include <inttypes.h>

class HammingWindowCommand
{
private:
    float *m_coefficients;
    int m_window_size;

public:
    HammingWindowCommand(int window_size);
    ~HammingWindowCommand();
    void applyWindowCommand(float *input);
};