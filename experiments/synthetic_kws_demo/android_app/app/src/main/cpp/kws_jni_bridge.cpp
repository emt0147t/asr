/**
 * JNI Bridge for Vietnamese Keyword Spotting (KWS) Mobile Audio Frontend.
 * Binds the native C++ Log-Mel preprocessing pipeline to Android/Kotlin.
 *
 * Package: com.example.synthetic_kws
 * Class:   KwsFrontendJni
 */

#include <jni.h>
#include <android/log.h>
#include <string>

#include "kws_mobile_frontend.h"

#define LOG_TAG "KwsFrontendJni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

/**
 * Initializes frontend lookup tables (Periodic Hann window, FFT twiddle factors).
 * Thread-safe and idempotent.
 *
 * Signature: ()I
 */
JNIEXPORT jint JNICALL
Java_com_example_synthetic_1kws_KwsFrontendJni_init(
    JNIEnv* /* env */,
    jobject /* thiz */) {
    int status = kws_frontend_init();
    if (status != KWS_SUCCESS) {
        LOGE("Failed to initialize KWS frontend: %d", status);
    }
    return static_cast<jint>(status);
}

/**
 * Computes 40-by-101 Log-Mel feature spectrogram from raw float32 PCM audio.
 *
 * Signature: ([FI[F)I
 *
 * @param audioIn     FloatArray containing exactly 16,000 PCM samples in [-1.0, 1.0].
 * @param numSamples  Must equal 16000.
 * @param featuresOut FloatArray of at least 4,040 elements to receive [40][101] Log-Mel features.
 * @return 0 on success, negative error code on failure.
 */
JNIEXPORT jint JNICALL
Java_com_example_synthetic_1kws_KwsFrontendJni_computeLogMel(
    JNIEnv* env,
    jobject /* thiz */,
    jfloatArray audioIn,
    jint numSamples,
    jfloatArray featuresOut) {

    // 1. Argument Null Checks
    if (audioIn == nullptr || featuresOut == nullptr) {
        LOGE("kws_frontend_compute failed: Null pointer argument provided.");
        return static_cast<jint>(KWS_ERR_NULL_POINTER);
    }

    // 2. Buffer Dimension Validation
    jsize inLen = env->GetArrayLength(audioIn);
    jsize outLen = env->GetArrayLength(featuresOut);

    if (numSamples != KWS_INPUT_SAMPLES || inLen < KWS_INPUT_SAMPLES) {
        LOGE("kws_frontend_compute failed: Expected %d samples, got numSamples=%d, inLen=%d",
             KWS_INPUT_SAMPLES, (int)numSamples, (int)inLen);
        return static_cast<jint>(KWS_ERR_INVALID_SAMPLES);
    }

    if (outLen < KWS_FEATURE_ELEMENTS) {
        LOGE("kws_frontend_compute failed: Output buffer undersized. Expected at least %d, got %d",
             KWS_FEATURE_ELEMENTS, (int)outLen);
        return static_cast<jint>(KWS_ERR_INVALID_SAMPLES);
    }

    // 3. Pin Native Array Buffers
    jfloat* audioPtr = env->GetFloatArrayElements(audioIn, nullptr);
    if (audioPtr == nullptr) {
        LOGE("Failed to acquire float array elements for audioIn.");
        return static_cast<jint>(KWS_ERR_NULL_POINTER);
    }

    jfloat* featuresPtr = env->GetFloatArrayElements(featuresOut, nullptr);
    if (featuresPtr == nullptr) {
        env->ReleaseFloatArrayElements(audioIn, audioPtr, JNI_ABORT);
        LOGE("Failed to acquire float array elements for featuresOut.");
        return static_cast<jint>(KWS_ERR_NULL_POINTER);
    }

    // 4. Execute C++ Frontend Extraction
    int status = kws_frontend_compute(audioPtr, numSamples, featuresPtr);

    // 5. Release Arrays (audioIn is read-only -> JNI_ABORT; featuresOut is written -> 0)
    env->ReleaseFloatArrayElements(audioIn, audioPtr, JNI_ABORT);
    env->ReleaseFloatArrayElements(featuresOut, featuresPtr, 0);

    if (status != KWS_SUCCESS) {
        LOGE("kws_frontend_compute execution failed with code %d: %s",
             status, kws_frontend_get_error_string(status));
    }

    return static_cast<jint>(status);
}

/**
 * Returns human-readable error description for a frontend error code.
 *
 * Signature: (I)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_example_synthetic_1kws_KwsFrontendJni_getErrorString(
    JNIEnv* env,
    jobject /* thiz */,
    jint errorCode) {
    const char* desc = kws_frontend_get_error_string(static_cast<int>(errorCode));
    return env->NewStringUTF(desc);
}

} // extern "C"
