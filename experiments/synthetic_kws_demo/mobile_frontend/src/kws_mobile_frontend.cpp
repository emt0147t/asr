/**
 * Portable C/C++ Audio Preprocessing Frontend Implementation.
 * Bit-accurate recreation of Python LogMelFeatureExtractor.
 */

#ifndef KWS_FRONTEND_BUILD_DLL
#define KWS_FRONTEND_BUILD_DLL
#endif

#include "kws_mobile_frontend.h"
#include "mel_filterbank_40x257.h"

#include <cmath>
#include <cstring>
#include <atomic>

/* Static lookup tables */
static float g_window_padded[KWS_FFT_SIZE];
static float g_twiddle_cos[KWS_FFT_SIZE];
static float g_twiddle_sin[KWS_FFT_SIZE];
static int   g_bit_rev[KWS_FFT_SIZE];
static std::atomic<bool> g_is_initialized(false);

static const double PI = 3.14159265358979323846;

int kws_frontend_init(void) {
    if (g_is_initialized.load(std::memory_order_acquire)) {
        return KWS_SUCCESS;
    }

    /* 1. Periodic Hann Window padded to 512 (56 leading zeros, 56 trailing zeros) */
    std::memset(g_window_padded, 0, sizeof(g_window_padded));
    for (int n = 0; n < KWS_WIN_LENGTH; n++) {
        /* Periodic Hann formula: 0.5 - 0.5 * cos(2 * pi * n / 400.0) */
        double w = 0.5 - 0.5 * std::cos((2.0 * PI * (double)n) / (double)KWS_WIN_LENGTH);
        g_window_padded[56 + n] = (float)w;
    }

    /* 2. FFT Twiddle factors table: W_N^k = exp(-2j * pi * k / 512) */
    for (int k = 0; k < KWS_FFT_SIZE; k++) {
        double angle = (-2.0 * PI * (double)k) / (double)KWS_FFT_SIZE;
        g_twiddle_cos[k] = (float)std::cos(angle);
        g_twiddle_sin[k] = (float)std::sin(angle);
    }

    /* 3. 9-bit bit-reversal lookup table for length 512 */
    for (int i = 0; i < KWS_FFT_SIZE; i++) {
        int rev = 0;
        int val = i;
        for (int b = 0; b < 9; b++) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        g_bit_rev[i] = rev;
    }

    g_is_initialized.store(true, std::memory_order_release);
    return KWS_SUCCESS;
}

int kws_fft_rfft512(const float* frame_in, float* real_out, float* imag_out) {
    if (!frame_in || !real_out || !imag_out) {
        return KWS_ERR_NULL_POINTER;
    }

    if (!g_is_initialized.load(std::memory_order_acquire)) {
        kws_frontend_init();
    }

    float xr[KWS_FFT_SIZE];
    float xi[KWS_FFT_SIZE];

    /* Bit-reversal permutation */
    for (int i = 0; i < KWS_FFT_SIZE; i++) {
        xr[i] = frame_in[g_bit_rev[i]];
        xi[i] = 0.0f;
    }

    /* Cooley-Tukey Radix-2 in-place DIT FFT */
    for (int length = 2; length <= KWS_FFT_SIZE; length <<= 1) {
        int half = length >> 1;
        int step = KWS_FFT_SIZE / length;

        for (int i = 0; i < KWS_FFT_SIZE; i += length) {
            for (int j = 0; j < half; j++) {
                int k = j * step;
                float c = g_twiddle_cos[k];
                float s = g_twiddle_sin[k];

                int idx1 = i + j;
                int idx2 = idx1 + half;

                float tr = xr[idx2] * c - xi[idx2] * s;
                float ti = xr[idx2] * s + xi[idx2] * c;

                xr[idx2] = xr[idx1] - tr;
                xi[idx2] = xi[idx1] - ti;
                xr[idx1] += tr;
                xi[idx1] += ti;
            }
        }
    }

    /* Return one-sided complex spectrum: bins 0 to 256 inclusive */
    for (int k = 0; k < KWS_FFT_BINS; k++) {
        real_out[k] = xr[k];
        imag_out[k] = xi[k];
    }

    return KWS_SUCCESS;
}

int kws_frontend_compute(const float* audio_in, int num_samples, float* features_out) {
    /* 1. Input Validation */
    if (!audio_in || !features_out) {
        return KWS_ERR_NULL_POINTER;
    }
    if (num_samples != KWS_INPUT_SAMPLES) {
        return KWS_ERR_INVALID_SAMPLES;
    }

    /* Check for non-finite values */
    for (int i = 0; i < num_samples; i++) {
        float s = audio_in[i];
        if (std::isnan(s) || std::isinf(s)) {
            return KWS_ERR_NON_FINITE_INPUT;
        }
    }

    if (!g_is_initialized.load(std::memory_order_acquire)) {
        kws_frontend_init();
    }

    /* 2. Symmetric Reflect Padding: pad by 256 samples on both ends */
    float padded[KWS_PADDED_SAMPLES];

    /* Left reflection: padded[256 - k] = audio[k] for k = 1..256 */
    for (int k = 1; k <= KWS_PAD_REFLECT; k++) {
        padded[KWS_PAD_REFLECT - k] = audio_in[k];
    }

    /* Center: copy original 16,000 samples */
    std::memcpy(&padded[KWS_PAD_REFLECT], audio_in, KWS_INPUT_SAMPLES * sizeof(float));

    /* Right reflection: padded[16256 + k - 1] = audio[15999 - k] for k = 1..256 */
    int center_end = KWS_PAD_REFLECT + KWS_INPUT_SAMPLES;
    for (int k = 1; k <= KWS_PAD_REFLECT; k++) {
        padded[center_end + k - 1] = audio_in[KWS_INPUT_SAMPLES - 1 - k];
    }

    /* Temporary frame and spectrum buffers */
    float frame[KWS_FFT_SIZE];
    float real[KWS_FFT_BINS];
    float imag[KWS_FFT_BINS];
    float power[KWS_FFT_BINS];

    /* 3. STFT Framing across 101 frames */
    for (int m = 0; m < KWS_TIME_FRAMES; m++) {
        int frame_start = m * KWS_HOP_LENGTH;

        /* Windowing: only indices 56..455 are non-zero in g_window_padded */
        std::memset(frame, 0, sizeof(frame));
        for (int i = 56; i < 456; i++) {
            frame[i] = padded[frame_start + i] * g_window_padded[i];
        }

        /* 512-point Real FFT */
        kws_fft_rfft512(frame, real, imag);

        /* Power Spectrum: |X|^2 = re^2 + im^2 (power = 2.0) */
        for (int k = 0; k < KWS_FFT_BINS; k++) {
            power[k] = real[k] * real[k] + imag[k] * imag[k];
        }

        /* 4. Mel Filterbank Dot Product & Natural Logarithm */
        for (int b = 0; b < KWS_MEL_BANDS; b++) {
            double mel_power = 0.0;
            const float* filter_row = KWS_MEL_FILTERBANK[b];

            for (int k = 0; k < KWS_FFT_BINS; k++) {
                mel_power += (double)filter_row[k] * (double)power[k];
            }

            /* Log compression: ln(mel_power + 1e-6) */
            double log_val = std::log(mel_power + (double)KWS_LOG_EPSILON);

            /* Row-major output: [40][101] -> features_out[b * 101 + m] */
            features_out[b * KWS_TIME_FRAMES + m] = (float)log_val;
        }
    }

    return KWS_SUCCESS;
}

const char* kws_frontend_get_error_string(int error_code) {
    switch (error_code) {
        case KWS_SUCCESS:
            return "Success";
        case KWS_ERR_NULL_POINTER:
            return "Null pointer argument provided";
        case KWS_ERR_INVALID_SAMPLES:
            return "Invalid sample count (expected exactly 16,000 samples / 1.000s)";
        case KWS_ERR_NON_FINITE_INPUT:
            return "Audio contains non-finite floating-point values (NaN or Inf)";
        case KWS_ERR_UNINITIALIZED:
            return "Frontend static tables uninitialized";
        default:
            return "Unknown error";
    }
}
