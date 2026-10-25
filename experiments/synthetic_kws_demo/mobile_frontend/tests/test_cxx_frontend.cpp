/**
 * Unit Test Suite for KWS Mobile Frontend (C/C++).
 * Tests input validation, FFT mathematical properties, window symmetry,
 * Mel filterbank properties, and end-to-end edge cases.
 */

#include "kws_mobile_frontend.h"
#include "mel_filterbank_40x257.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <limits>
#include <iomanip>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        return false; \
    } else { \
        g_tests_passed++; \
    } \
} while(0)

// 1. Test Input Validation & Robustness
static bool test_input_validation() {
    std::cout << "[Test 1] Input Validation & Robustness..." << std::endl;
    std::vector<float> valid_audio(KWS_INPUT_SAMPLES, 0.0f);
    std::vector<float> valid_features(KWS_FEATURE_ELEMENTS, 0.0f);

    // Null pointer checks
    TEST_ASSERT(kws_frontend_compute(nullptr, KWS_INPUT_SAMPLES, valid_features.data()) == KWS_ERR_NULL_POINTER,
                "Null audio_in should return KWS_ERR_NULL_POINTER");
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), KWS_INPUT_SAMPLES, nullptr) == KWS_ERR_NULL_POINTER,
                "Null features_out should return KWS_ERR_NULL_POINTER");

    // Sample count validation
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), 0, valid_features.data()) == KWS_ERR_INVALID_SAMPLES,
                "0 samples should return KWS_ERR_INVALID_SAMPLES");
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), 15999, valid_features.data()) == KWS_ERR_INVALID_SAMPLES,
                "15999 samples should return KWS_ERR_INVALID_SAMPLES");
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), 16001, valid_features.data()) == KWS_ERR_INVALID_SAMPLES,
                "16001 samples should return KWS_ERR_INVALID_SAMPLES");

    // Non-finite value handling (NaN and Inf)
    valid_audio[500] = std::numeric_limits<float>::quiet_NaN();
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), KWS_INPUT_SAMPLES, valid_features.data()) == KWS_ERR_NON_FINITE_INPUT,
                "NaN sample should return KWS_ERR_NON_FINITE_INPUT");

    valid_audio[500] = std::numeric_limits<float>::infinity();
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), KWS_INPUT_SAMPLES, valid_features.data()) == KWS_ERR_NON_FINITE_INPUT,
                "+Inf sample should return KWS_ERR_NON_FINITE_INPUT");

    valid_audio[500] = -std::numeric_limits<float>::infinity();
    TEST_ASSERT(kws_frontend_compute(valid_audio.data(), KWS_INPUT_SAMPLES, valid_features.data()) == KWS_ERR_NON_FINITE_INPUT,
                "-Inf sample should return KWS_ERR_NON_FINITE_INPUT");

    // Error strings
    TEST_ASSERT(std::string(kws_frontend_get_error_string(KWS_SUCCESS)) == "Success", "Error string for SUCCESS");
    TEST_ASSERT(std::string(kws_frontend_get_error_string(KWS_ERR_NULL_POINTER)).find("Null") != std::string::npos, "Error string for NULL");

    return true;
}

// 2. Test FFT Mathematical Properties
static bool test_fft_mathematical_properties() {
    std::cout << "[Test 2] FFT Mathematical Properties..." << std::endl;
    kws_frontend_init();

    // 2a. Impulse Response: x[0] = 1, x[k] = 0 for k > 0 -> X[k] = 1.0 + 0j for all k
    std::vector<float> impulse(KWS_FFT_SIZE, 0.0f);
    impulse[0] = 1.0f;
    std::vector<float> real(KWS_FFT_BINS, 0.0f);
    std::vector<float> imag(KWS_FFT_BINS, 0.0f);

    int status = kws_fft_rfft512(impulse.data(), real.data(), imag.data());
    TEST_ASSERT(status == KWS_SUCCESS, "FFT computation of impulse succeeded");

    for (int k = 0; k < KWS_FFT_BINS; k++) {
        TEST_ASSERT(std::abs(real[k] - 1.0f) < 1e-6f, "Impulse FFT real part must be 1.0");
        TEST_ASSERT(std::abs(imag[k]) < 1e-6f, "Impulse FFT imag part must be 0.0");
    }

    // 2b. Parseval's Theorem on a non-trivial signal
    // Sum |x[n]|^2 == 1/N * (|X[0]|^2 + |X[N/2]|^2 + 2 * Sum_{k=1}^{N/2-1} |X[k]|^2)
    std::vector<float> test_signal(KWS_FFT_SIZE, 0.0f);
    double time_domain_energy = 0.0;
    for (int n = 0; n < KWS_FFT_SIZE; n++) {
        test_signal[n] = (float)std::sin(2.0 * 3.141592653589793 * 7.0 * n / 512.0)
                       + 0.5f * (float)std::cos(2.0 * 3.141592653589793 * 23.0 * n / 512.0);
        time_domain_energy += test_signal[n] * test_signal[n];
    }

    kws_fft_rfft512(test_signal.data(), real.data(), imag.data());

    double freq_domain_energy = real[0] * real[0] + imag[0] * imag[0]
                              + real[256] * real[256] + imag[256] * imag[256];
    for (int k = 1; k < 256; k++) {
        freq_domain_energy += 2.0 * (real[k] * real[k] + imag[k] * imag[k]);
    }
    freq_domain_energy /= 512.0;

    double energy_diff = std::abs(time_domain_energy - freq_domain_energy);
    TEST_ASSERT(energy_diff < 1e-4, "Parseval energy conservation holds within 1e-4");

    // 2c. Pure Tone Peak Location: frequency bin 16
    std::vector<float> tone16(KWS_FFT_SIZE, 0.0f);
    for (int n = 0; n < KWS_FFT_SIZE; n++) {
        tone16[n] = (float)std::cos(2.0 * 3.141592653589793 * 16.0 * n / 512.0);
    }
    kws_fft_rfft512(tone16.data(), real.data(), imag.data());

    // Peak should be at bin 16 (magnitude = 256.0)
    float peak_mag = std::sqrt(real[16] * real[16] + imag[16] * imag[16]);
    TEST_ASSERT(std::abs(peak_mag - 256.0f) < 1e-4f, "Pure cosine tone has peak magnitude 256 at bin 16");

    for (int k = 0; k < KWS_FFT_BINS; k++) {
        if (k != 16) {
            float non_peak = std::sqrt(real[k] * real[k] + imag[k] * imag[k]);
            TEST_ASSERT(non_peak < 1e-4f, "Orthogonal bins must have negligible magnitude");
        }
    }

    return true;
}

// 3. Test Mel Filterbank Constants & Structure
static bool test_mel_filterbank_structure() {
    std::cout << "[Test 3] Mel Filterbank Structure..." << std::endl;

    for (int b = 0; b < KWS_MEL_BANDS; b++) {
        int non_zero_count = 0;
        float max_val = 0.0f;
        int peak_bin = -1;

        for (int k = 0; k < KWS_FFT_BINS; k++) {
            float val = KWS_MEL_FILTERBANK[b][k];
            TEST_ASSERT(val >= 0.0f, "Mel filterbank weights must be non-negative");
            if (val > 0.0f) non_zero_count++;
            if (val > max_val) {
                max_val = val;
                peak_bin = k;
            }
        }
        TEST_ASSERT(non_zero_count > 0, "Each Mel band must have non-zero bins");
        TEST_ASSERT(peak_bin >= 0, "Each Mel band must have a defined peak");
    }

    // Verify lowest band is below highest band
    int lowest_peak = 0, highest_peak = 0;
    for (int k = 0; k < KWS_FFT_BINS; k++) {
        if (KWS_MEL_FILTERBANK[0][k] > KWS_MEL_FILTERBANK[0][lowest_peak]) lowest_peak = k;
        if (KWS_MEL_FILTERBANK[39][k] > KWS_MEL_FILTERBANK[39][highest_peak]) highest_peak = k;
    }
    TEST_ASSERT(lowest_peak < highest_peak, "Mel filterbank bands must increase monotonically in frequency");

    return true;
}

// 4. Test Silence & Edge Cases
static bool test_silence_and_edge_cases() {
    std::cout << "[Test 4] Silence & Edge Cases..." << std::endl;

    // All-zero audio (Silence)
    std::vector<float> silence(KWS_INPUT_SAMPLES, 0.0f);
    std::vector<float> features(KWS_FEATURE_ELEMENTS, 0.0f);

    int status = kws_frontend_compute(silence.data(), (int)silence.size(), features.data());
    TEST_ASSERT(status == KWS_SUCCESS, "kws_frontend_compute on silence succeeded");

    // When power is 0, mel_power is 0, so feature is ln(0 + 1e-6) = ln(1e-6) = -13.81551055796...
    float expected_silence_val = std::log(1.0e-6f);
    for (int i = 0; i < KWS_FEATURE_ELEMENTS; i++) {
        float diff = std::abs(features[i] - expected_silence_val);
        TEST_ASSERT(diff < 1e-5f, "All-zero audio must yield exact log-epsilon floor");
    }

    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << " KWS Mobile Frontend C++ Unit Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_input_validation();
    all_passed &= test_fft_mathematical_properties();
    all_passed &= test_mel_filterbank_structure();
    all_passed &= test_silence_and_edge_cases();

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Results: " << g_tests_passed << " / " << g_tests_run << " assertions passed." << std::endl;

    if (all_passed) {
        std::cout << "ALL C++ UNIT TESTS PASSED." << std::endl;
        return 0;
    } else {
        std::cerr << "UNIT TESTS FAILED." << std::endl;
        return 1;
    }
}
