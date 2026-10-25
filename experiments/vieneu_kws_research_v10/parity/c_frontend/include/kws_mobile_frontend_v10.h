/**
 * Portable C/C++ Audio Preprocessing Frontend for Vietnamese KWS (Phase 92).
 * Canonical Candidate: variant_w10 (CorrectedVietnameseKWSNet)
 * Matches bit-level and mathematical specifications of Torchaudio LogMelFrontend:
 *   - Sample rate: 16,000 Hz mono PCM float32
 *   - Input buffer: exactly 16,000 samples (1.000s)
 *   - Window: Periodic Hann (400 samples)
 *   - STFT: 400-point real FFT, hop 160, reflect-padded by 200 samples (101 frames)
 *   - Mel filterbank: 40 Slaney area-normalized triangular filters (20 - 8,000 Hz)
 *   - Log compression: natural logarithm ln(mel_power + 1e-6)
 *   - Output layout: [40][101] row-major float32 (Batch=1, Channels=1, Mel=40, Time=101)
 *
 * Dependency-free: standard C/C++ math library.
 */

#ifndef KWS_MOBILE_FRONTEND_V10_H
#define KWS_MOBILE_FRONTEND_V10_H

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

/* Canonical constants matching Python LogMelFrontend (variant_w10) */
#define KWS_V10_SAMPLE_RATE          16000
#define KWS_V10_INPUT_SAMPLES        16000
#define KWS_V10_MEL_BANDS            40
#define KWS_V10_TIME_FRAMES          101
#define KWS_V10_FEATURE_ELEMENTS     4040   /* 40 * 101 */
#define KWS_V10_FFT_SIZE             400
#define KWS_V10_WIN_LENGTH           400
#define KWS_V10_HOP_LENGTH           160
#define KWS_V10_PAD_REFLECT          200
#define KWS_V10_PADDED_SAMPLES       16400  /* 16000 + 200 + 200 */
#define KWS_V10_FFT_BINS             201    /* 400 / 2 + 1 */
#define KWS_V10_LOG_EPSILON          1.0e-6f

/* Return status and error codes */
#define KWS_V10_SUCCESS               0
#define KWS_V10_ERR_NULL_POINTER     -1
#define KWS_V10_ERR_INVALID_SAMPLES  -2
#define KWS_V10_ERR_NON_FINITE_INPUT -3
#define KWS_V10_ERR_UNINITIALIZED    -4

/**
 * Initializes global static tables (Hann window, DFT twiddle factors).
 * Thread-safe and idempotent.
 * @return KWS_V10_SUCCESS (0).
 */
KWS_API int kws_frontend_v10_init(void);

/**
 * Computes 40-by-101 Log-Mel feature spectrogram from raw float32 audio.
 * Peak clamps if amplitude exceeds 1.0 (never amplifies quiet signals).
 * @param audio_in Pointer to 16,000 float32 audio samples in [-1.0, 1.0].
 * @param num_samples Must be exactly 16,000.
 * @param features_out Pointer to destination buffer of at least 4,040 float32 elements.
 *                     Memory layout is row-major: features_out[mel_band * 101 + frame_idx].
 * @return KWS_V10_SUCCESS (0) on success, or negative error code.
 */
KWS_API int kws_frontend_v10_compute(const float* audio_in, int num_samples, float* features_out);

/**
 * Computes single 400-point real DFT.
 * @param frame_in Pointer to 400 float32 windowed input samples.
 * @param real_out Destination for 201 float32 real parts.
 * @param imag_out Destination for 201 float32 imaginary parts.
 * @return KWS_V10_SUCCESS (0).
 */
KWS_API int kws_dft_rfft400(const float* frame_in, float* real_out, float* imag_out);

/**
 * Returns descriptive error string for an error code.
 */
KWS_API const char* kws_frontend_v10_get_error_string(int error_code);

#ifdef __cplusplus
}
#endif

#endif /* KWS_MOBILE_FRONTEND_V10_H */
