package com.example.synthetic_kws

/**
 * Kotlin wrapper for native C++ Log-Mel audio frontend library.
 * Loads 'kws_mobile_frontend_jni' and exposes JNI native methods.
 */
object KwsFrontendJni {

    private var isLibraryLoaded = false

    init {
        try {
            System.loadLibrary("kws_mobile_frontend_jni")
            isLibraryLoaded = true
        } catch (e: UnsatisfiedLinkError) {
            // In unit tests or mock environments without the native .so, mark as not loaded
            isLibraryLoaded = false
        }
    }

    fun isNativeLoaded(): Boolean = isLibraryLoaded

    /**
     * Initializes frontend tables (Periodic Hann window, FFT twiddle factors).
     * @return 0 on success, negative error code on failure.
     */
    external fun init(): Int

    /**
     * Computes 40-by-101 Log-Mel features from 16,000 PCM float32 samples.
     * @param audioIn Exactly 16,000 float32 samples in [-1.0, 1.0].
     * @param numSamples Must equal 16,000.
     * @param featuresOut Buffer of at least 4,040 floats to receive output.
     * @return 0 on success, negative error code on failure.
     */
    external fun computeLogMel(audioIn: FloatArray, numSamples: Int, featuresOut: FloatArray): Int

    /**
     * Returns human-readable error description for a native status code.
     */
    external fun getErrorString(errorCode: Int): String

    const val KWS_SUCCESS = 0
    const val KWS_ERR_NULL_POINTER = -1
    const val KWS_ERR_INVALID_SAMPLES = -2
    const val KWS_ERR_NON_FINITE_INPUT = -3
    const val KWS_ERR_UNINITIALIZED = -4

    const val SAMPLE_RATE = 16000
    const val INPUT_SAMPLES = 16000
    const val MEL_BANDS = 40
    const val TIME_FRAMES = 101
    const val TOTAL_FEATURES = 4040
}
