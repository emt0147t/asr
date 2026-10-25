/**
 * Standalone CLI Runner for KWS Mobile Frontend.
 * Reads a 16kHz mono WAV (16-bit PCM or 32-bit float) or raw float32 file,
 * computes 40x101 Log-Mel spectrogram, and writes raw binary float32 tensor.
 *
 * Usage:
 *   run_frontend_cli <input_audio.wav|raw> <output_features.bin>
 *   run_frontend_cli --input <file> --output <file>
 */

#include "kws_mobile_frontend.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <cmath>

#pragma pack(push, 1)
struct WavRiffHeader {
    char chunk_id[4];     // "RIFF"
    uint32_t chunk_size;
    char format[4];       // "WAVE"
};

struct WavChunkHeader {
    char subchunk_id[4];
    uint32_t subchunk_size;
};

struct WavFmtChunk {
    uint16_t audio_format;    // 1 = PCM, 3 = IEEE float
    uint16_t num_channels;    // 1 = mono
    uint32_t sample_rate;     // 16000
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample; // 16 or 32
};
#pragma pack(pop)

static bool read_wav_or_raw(const std::string& path, std::vector<float>& audio_out) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file: " << path << std::endl;
        return false;
    }

    // Check for WAV magic "RIFF"
    char magic[4] = {0};
    file.read(magic, 4);
    if (file.gcount() == 4 && std::strncmp(magic, "RIFF", 4) == 0) {
        // It's a WAV file
        file.seekg(0, std::ios::beg);
        WavRiffHeader riff;
        if (!file.read(reinterpret_cast<char*>(&riff), sizeof(riff))) {
            std::cerr << "Error: Corrupted RIFF header: " << path << std::endl;
            return false;
        }

        if (std::strncmp(riff.format, "WAVE", 4) != 0) {
            std::cerr << "Error: Not a WAVE format file." << std::endl;
            return false;
        }

        WavFmtChunk fmt;
        bool fmt_found = false;
        std::vector<char> raw_data;

        while (file) {
            WavChunkHeader chunk;
            if (!file.read(reinterpret_cast<char*>(&chunk), sizeof(chunk))) {
                break;
            }

            if (std::strncmp(chunk.subchunk_id, "fmt ", 4) == 0) {
                if (!file.read(reinterpret_cast<char*>(&fmt), std::min(sizeof(fmt), (size_t)chunk.subchunk_size))) {
                    std::cerr << "Error: Failed to read fmt chunk." << std::endl;
                    return false;
                }
                if (chunk.subchunk_size > sizeof(fmt)) {
                    file.seekg(chunk.subchunk_size - sizeof(fmt), std::ios::cur);
                }
                fmt_found = true;
            } else if (std::strncmp(chunk.subchunk_id, "data", 4) == 0) {
                raw_data.resize(chunk.subchunk_size);
                if (!file.read(raw_data.data(), chunk.subchunk_size)) {
                    std::cerr << "Error: Failed to read data chunk." << std::endl;
                    return false;
                }
                break;
            } else {
                // Skip unknown chunk
                file.seekg(chunk.subchunk_size, std::ios::cur);
            }
        }

        if (!fmt_found || raw_data.empty()) {
            std::cerr << "Error: WAV file missing fmt or data chunk: " << path << std::endl;
            return false;
        }

        if (fmt.num_channels != 1 || fmt.sample_rate != 16000) {
            std::cerr << "Error: Input WAV must be 16,000 Hz mono (channels=" 
                      << fmt.num_channels << ", sample_rate=" << fmt.sample_rate << ")." << std::endl;
            return false;
        }

        if (fmt.audio_format == 1 && fmt.bits_per_sample == 16) {
            size_t num_samples = raw_data.size() / sizeof(int16_t);
            const int16_t* pcm16 = reinterpret_cast<const int16_t*>(raw_data.data());
            audio_out.resize(num_samples);
            for (size_t i = 0; i < num_samples; i++) {
                audio_out[i] = static_cast<float>(pcm16[i]) / 32768.0f;
            }
        } else if ((fmt.audio_format == 3 || fmt.audio_format == 1) && fmt.bits_per_sample == 32) {
            size_t num_samples = raw_data.size() / sizeof(float);
            const float* pcm32 = reinterpret_cast<const float*>(raw_data.data());
            audio_out.assign(pcm32, pcm32 + num_samples);
        } else {
            std::cerr << "Error: Unsupported WAV format (format=" << fmt.audio_format 
                      << ", bits=" << fmt.bits_per_sample << ")." << std::endl;
            return false;
        }
    } else {
        // Assume raw float32 file
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t num_samples = file_size / sizeof(float);
        audio_out.resize(num_samples);
        if (!file.read(reinterpret_cast<char*>(audio_out.data()), file_size)) {
            std::cerr << "Error: Failed reading raw float32 file." << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (input_path.empty()) {
            input_path = arg;
        } else if (output_path.empty()) {
            output_path = arg;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Usage: run_frontend_cli <input_audio.wav|raw> <output_features.bin>" << std::endl;
        std::cerr << "   or: run_frontend_cli --input <file> --output <file>" << std::endl;
        return 1;
    }

    std::vector<float> audio;
    if (!read_wav_or_raw(input_path, audio)) {
        return 2;
    }

    if (audio.size() != KWS_INPUT_SAMPLES) {
        std::cerr << "Error: Expected exactly " << KWS_INPUT_SAMPLES 
                  << " samples, but found " << audio.size() << " samples." << std::endl;
        return 3;
    }

    // Allocate 40x101 features buffer
    std::vector<float> features(KWS_FEATURE_ELEMENTS, 0.0f);

    int status = kws_frontend_compute(audio.data(), (int)audio.size(), features.data());
    if (status != KWS_SUCCESS) {
        std::cerr << "Error: kws_frontend_compute failed with code " 
                  << status << " (" << kws_frontend_get_error_string(status) << ")" << std::endl;
        return 4;
    }

    // Compute basic statistics
    float min_val = features[0], max_val = features[0];
    double sum_val = 0.0;
    for (float v : features) {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        sum_val += v;
    }
    double mean_val = sum_val / (double)features.size();

    // Write binary output
    std::ofstream out_file(output_path, std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "Error: Could not open output file for writing: " << output_path << std::endl;
        return 5;
    }
    out_file.write(reinterpret_cast<const char*>(features.data()), features.size() * sizeof(float));
    out_file.close();

    std::cout << "KWS Mobile Frontend successfully processed: " << input_path << std::endl;
    std::cout << "  Output: " << output_path << " (" << features.size() * sizeof(float) << " bytes)" << std::endl;
    std::cout << "  Tensor shape: [40, 101] (4040 float32 elements)" << std::endl;
    std::cout << "  Min: " << min_val << ", Max: " << max_val << ", Mean: " << mean_val << std::endl;

    return 0;
}
