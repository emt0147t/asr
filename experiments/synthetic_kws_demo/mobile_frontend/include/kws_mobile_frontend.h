/**
 * Portable C/C++ Audio Preprocessing Frontend for Vietnamese KWS.
 * Matches bit-level and mathematical specifications of PyTorch LogMelFeatureExtractor:
 *   - Sample rate: 16,000 Hz mono PCM float32
 *   - Input buffer: exactly 16,000 samples (1.000s)
 *   - Window: Periodic Hann (400 samples, padded with 56 leading and 56 trailing zeros)
 *   - STFT: 512-point FFT, hop 160, reflect-padded by 256 samples (101 frames)
 *   - Mel filterbank: 40 Slaney area-normalized triangular filters (20 - 8,000 Hz)
 *   - Log compression: natural logarithm ln(mel_power + 1e-6)
 *   - Output layout: [40][101] row-major float32 (Batch=1, Channels=1, Mel=40, Time=101)
 *
 * Dependency-free: standard C/C++ math library (math.h / cmath).
 * Thread-safe: reentrant, read-only constant tables.
 */

#ifndef KWS_MOBILE_FRONTEND_H
#define KWS_MOBILE_FRONTEND_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(KWS_FRONTEND_BUILD_DLL)
    #define KWS_API __declspec(dllexport)
  #else
    #define KWS_API
  #endif
#else
  #define KWS_API __attribute__((visibility("default")))
#endif

/* Canonical constants matching Python LogMelFeatureExtractor */
#define KWS_SAMPLE_RATE          16000
#define KWS_INPUT_SAMPLES        16000
#define KWS_MEL_BANDS            40
#define KWS_TIME_FRAMES          101
#define KWS_FEATURE_ELEMENTS     4040   /* 40 * 101 */
#define KWS_FFT_SIZE             512
#define KWS_WIN_LENGTH           400
#define KWS_HOP_LENGTH           160
#define KWS_PAD_REFLECT          256
#define KWS_PADDED_SAMPLES       16512  /* 16000 + 256 + 256 */
#define KWS_FFT_BINS             257    /* 512 / 2 + 1 */
#define KWS_LOG_EPSILON          1.0e-6f

/* Return status and error codes */
#define KWS_SUCCESS               0
#define KWS_ERR_NULL_POINTER     -1
#define KWS_ERR_INVALID_SAMPLES  -2
#define KWS_ERR_NON_FINITE_INPUT -3
#define KWS_ERR_UNINITIALIZED    -4

/**
 * Initializes global static tables (window, twiddle factors, bit-reversals).
 * Thread-safe and idempotent; automatically invoked if uninitialized.
 * @return KWS_SUCCESS (0).
 */
KWS_API int kws_frontend_init(void);

/**
 * Computes 40-by-101 Log-Mel feature spectrogram from raw float32 audio.
 * @param audio_in Pointer to 16,000 float32 audio samples in [-1.0, 1.0].
 * @param num_samples Must be exactly 16,000.
 * @param features_out Pointer to destination buffer of at least 4,040 float32 elements.
 *                     Memory layout is row-major: features_out[mel_band * 101 + frame_idx].
 * @return KWS_SUCCESS (0) on success, or negative error code.
 */
KWS_API int kws_frontend_compute(const float* audio_in, int num_samples, float* features_out);

/**
 * Computes single 512-point real-to-complex FFT.
 * Provided for isolated numerical unit testing against golden FFT vectors.
 * @param frame_in Pointer to 512 float32 real input samples.
 * @param real_out Destination for 257 float32 real parts.
 * @param imag_out Destination for 257 float32 imaginary parts.
 * @return KWS_SUCCESS (0).
 */
KWS_API int kws_fft_rfft512(const float* frame_in, float* real_out, float* imag_out);

/**
 * Returns descriptive error string for an error code.
 */
KWS_API const char* kws_frontend_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* KWS_MOBILE_FRONTEND_H */
