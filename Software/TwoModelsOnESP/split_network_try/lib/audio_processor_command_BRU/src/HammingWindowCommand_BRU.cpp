#include <stdlib.h>
#include <math.h>
#include "HammingWindowCommand_BRU.h"

HammingWindowCommand_BRU::HammingWindowCommand_BRU(int window_size)
{
    m_window_size = window_size;
    m_coefficients = static_cast<float *>(malloc(sizeof(float) * m_window_size));
    // create the constants for a hamming window
    const float arg = M_PI * 2.0 / window_size;
    for (int i = 0; i < window_size; i++)
    {
        float float_value = 0.5 - (0.5 * cos(arg * (i + 0.5)));
        // Scale it to fixed point and round it.
        m_coefficients[i] = float_value;
    }
}

HammingWindowCommand_BRU::~HammingWindowCommand_BRU()
{
    free(m_coefficients);
}

void HammingWindowCommand_BRU::applyWindowCommand_BRU(float *input)
{
    for (int i = 0; i < m_window_size; i++)
    {
        input[i] = input[i] * m_coefficients[i];
    }
}
