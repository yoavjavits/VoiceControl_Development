#ifndef AUDIO_PROCESSOR_COMMAND_BRU
#define AUDIO_PROCESSOR_COMMAND_BRU

#include <stdlib.h>
#include <stdint.h>
// #define FIXED_POINT (16)
#include "./kissfft/tools/kiss_fftr.h"

class HammingWindowCommand_BRU;

class RingBufferAccessor;

class AudioProcessorCommand_BRU
{
private:
    int m_audio_length;
    int m_window_size;
    int m_step_size;
    int m_pooling_size;
    size_t m_fft_size;
    float *m_fft_input;
    int m_energy_size;
    int m_pooled_energy_size;
    float *m_energy;
    kiss_fft_cpx *m_fft_output;
    kiss_fftr_cfg m_cfg;
    float m_smoothed_noise_floor;

    HammingWindowCommand_BRU *m_hamming_window;

    void get_spectrogram_segment_BRU(float *output_spectrogram_row);

public:
    AudioProcessorCommand_BRU(int audio_length, int window_size, int step_size, int pooling_size);
    ~AudioProcessorCommand_BRU();
    bool get_spectrogramCommand_BRU(RingBufferAccessor *reader, float *output_spectrogram);
};

#endif