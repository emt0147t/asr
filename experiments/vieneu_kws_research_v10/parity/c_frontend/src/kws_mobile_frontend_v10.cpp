/**
 * Portable C/C++ Audio Preprocessing Frontend for Vietnamese KWS (Phase 92).
 * Implementation of 400-point STFT and 40-band Slaney Mel spectrogram.
 */

#ifndef KWS_FRONTEND_BUILD_DLL
#define KWS_FRONTEND_BUILD_DLL
#endif

#include "kws_mobile_frontend_v10.h"
#include "mel_filterbank_40x201.h"

#include <cmath>
#include <cstring>
#include <atomic>

/* Static lookup tables */
static float g_window[KWS_V10_WIN_LENGTH];
static float g_dft_cos[KWS_V10_FFT_BINS][KWS_V10_FFT_SIZE];
static float g_dft_sin[KWS_V10_FFT_BINS][KWS_V10_FFT_SIZE];
static std::atomic<bool> g_is_initialized(false);

static const double PI = 3.14159265358979323846;

int kws_frontend_v10_init(void) {
    if (g_is_initialized.load(std::memory_order_acquire)) {
        return KWS_V10_SUCCESS;
    }

    /* 1. Periodic Hann Window (400 samples) */
    for (int n = 0; n < KWS_V10_WIN_LENGTH; n++) {
        double w = 0.5 - 0.5 * std::cos((2.0 * PI * (double)n) / (double)KWS_V10_WIN_LENGTH);
        g_window[n] = (float)w;
    }

    /* 2. Twiddle factor lookup tables for 400-point real DFT */
    for (int k = 0; k < KWS_V10_FFT_BINS; k++) {
        for (int n = 0; n < KWS_V10_FFT_SIZE; n++) {
            double angle = (2.0 * PI * (double)k * (double)n) / (double)KWS_V10_FFT_SIZE;
            g_dft_cos[k][n] = (float)std::cos(angle);
            g_dft_sin[k][n] = (float)(-std::sin(angle));
        }
    }

    g_is_initialized.store(true, std::memory_order_release);
    return KWS_V10_SUCCESS;
}

int kws_dft_rfft400(const float* frame_in, float* real_out, float* imag_out) {
    if (!frame_in || !real_out || !imag_out) {
        return KWS_V10_ERR_NULL_POINTER;
    }

    if (!g_is_initialized.load(std::memory_order_acquire)) {
        kws_frontend_v10_init();
    }

    for (int k = 0; k < KWS_V10_FFT_BINS; k++) {
        double sum_r = 0.0;
        double sum_i = 0.0;
        const float* cos_row = g_dft_cos[k];
        const float* sin_row = g_dft_sin[k];

        for (int n = 0; n < KWS_V10_FFT_SIZE; n++) {
            float s = frame_in[n];
            sum_r += (double)(s * cos_row[n]);
            sum_i += (double)(s * sin_row[n]);
        }
        real_out[k] = (float)sum_r;
        imag_out[k] = (float)sum_i;
    }

    return KWS_V10_SUCCESS;
}

int kws_frontend_v10_compute(const float* audio_in, int num_samples, float* features_out) {
    /* 1. Argument validation */
    if (!audio_in || !features_out) {
        return KWS_V10_ERR_NULL_POINTER;
    }
    if (num_samples != KWS_V10_INPUT_SAMPLES) {
        return KWS_V10_ERR_INVALID_SAMPLES;
    }

    /* 2. Check for non-finite values and determine peak amplitude */
    float max_abs = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float s = audio_in[i];
        if (std::isnan(s) || std::isinf(s)) {
            return KWS_V10_ERR_NON_FINITE_INPUT;
        }
        float a = std::fabs(s);
        if (a > max_abs) {
            max_abs = a;
        }
    }

    float norm_factor = (max_abs > 1.0f) ? (1.0f / max_abs) : 1.0f;

    if (!g_is_initialized.load(std::memory_order_acquire)) {
        kws_frontend_v10_init();
    }

    /* 3. Symmetric Reflect Padding (200 samples on both ends) */
    float padded[KWS_V10_PADDED_SAMPLES];

    /* Left reflection: padded[200 - k] = audio[k] for k = 1..200 */
    for (int k = 1; k <= KWS_V10_PAD_REFLECT; k++) {
        padded[KWS_V10_PAD_REFLECT - k] = audio_in[k] * norm_factor;
    }

    /* Center: 16,000 samples */
    for (int i = 0; i < KWS_V10_INPUT_SAMPLES; i++) {
        padded[KWS_V10_PAD_REFLECT + i] = audio_in[i] * norm_factor;
    }

    /* Right reflection: padded[16200 + k - 1] = audio[15999 - k] for k = 1..200 */
    int center_end = KWS_V10_PAD_REFLECT + KWS_V10_INPUT_SAMPLES;
    for (int k = 1; k <= KWS_V10_PAD_REFLECT; k++) {
        padded[center_end + k - 1] = audio_in[KWS_V10_INPUT_SAMPLES - 1 - k] * norm_factor;
    }

    /* Buffers for single frame */
    float frame[KWS_V10_FFT_SIZE];
    float real[KWS_V10_FFT_BINS];
    float imag[KWS_V10_FFT_BINS];
    float power[KWS_V10_FFT_BINS];

    /* 4. STFT Framing across 101 frames */
    for (int m = 0; m < KWS_V10_TIME_FRAMES; m++) {
        int frame_start = m * KWS_V10_HOP_LENGTH;

        /* Apply window */
        for (int n = 0; n < KWS_V10_WIN_LENGTH; n++) {
            frame[n] = padded[frame_start + n] * g_window[n];
        }

        /* 400-point real DFT */
        kws_dft_rfft400(frame, real, imag);

        /* Power spectrum: |X|^2 = re^2 + im^2 */
        for (int k = 0; k < KWS_V10_FFT_BINS; k++) {
            power[k] = real[k] * real[k] + imag[k] * imag[k];
        }

        /* Mel Filterbank Dot Product & Logarithm */
        for (int b = 0; b < KWS_V10_MEL_BANDS; b++) {
            double mel_power = 0.0;
            const float* filter_row = KWS_MEL_FILTERBANK_40X201[b];

            for (int k = 0; k < KWS_V10_FFT_BINS; k++) {
                float w = filter_row[k];
                if (w > 0.0f) {
                    mel_power += (double)(power[k] * w);
                }
            }

            /* Log compression: ln(mel_power + 1e-6) */
            float log_val = (float)std::log(mel_power + (double)KWS_V10_LOG_EPSILON);

            /* Row-major layout: [mel_band][frame_idx] = b * 101 + m */
            features_out[b * KWS_V10_TIME_FRAMES + m] = log_val;
        }
    }

    return KWS_V10_SUCCESS;
}

const char* kws_frontend_v10_get_error_string(int error_code) {
    switch (error_code) {
        case KWS_V10_SUCCESS:
            return "Success";
        case KWS_V10_ERR_NULL_POINTER:
            return "Null pointer provided";
        case KWS_V10_ERR_INVALID_SAMPLES:
            return "Invalid number of audio samples (expected 16,000)";
        case KWS_V10_ERR_NON_FINITE_INPUT:
            return "Input contains NaN or Inf values";
        case KWS_V10_ERR_UNINITIALIZED:
            return "Frontend uninitialized";
        default:
            return "Unknown error";
    }
}
