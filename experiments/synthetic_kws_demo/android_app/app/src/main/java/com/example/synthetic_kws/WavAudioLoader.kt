package com.example.synthetic_kws

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Strict WAV/PCM audio loader and validator for 16,000 Hz mono KWS fixtures.
 * Enforces zero tolerance for invalid audio, sample rate mismatch, multichannel inputs,
 * and duration deviation.
 *
 * GOVERNANCE: Pure synthetic fixtures only. No live microphone or streaming capture.
 * Prohibits resampling, AGC, peak normalization, or silent distortion.
 */
object WavAudioLoader {

    const val REQUIRED_SAMPLE_RATE = 16000
    const val REQUIRED_CHANNELS = 1
    const val REQUIRED_SAMPLES = 16000

    data class WavInfo(
        val sampleRate: Int,
        val channels: Int,
        val bitsPerSample: Int,
        val formatCode: Int,
        val totalSamples: Int,
        val durationSec: Float,
        val audioData: FloatArray
    )

    fun loadAndValidate(inputStream: InputStream): WavInfo {
        val bytes = inputStream.readBytes()
        if (bytes.size < 44) {
            throw IllegalArgumentException("Audio file is too small to contain a valid WAV header (${bytes.size} bytes)")
        }

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

        // 1. RIFF header check
        val riff = ByteArray(4)
        buffer.get(riff)
        if (String(riff) != "RIFF") {
            throw IllegalArgumentException("Invalid WAV header: Missing 'RIFF' magic signature")
        }

        val riffSize = buffer.int
        val wave = ByteArray(4)
        buffer.get(wave)
        if (String(wave) != "WAVE") {
            throw IllegalArgumentException("Invalid WAV format: Missing 'WAVE' identifier")
        }

        // 2. Scan chunks for 'fmt ' and 'data'
        var formatCode = 0
        var channels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var dataOffset = -1
        var dataSize = 0
        var fmtFound = false

        while (buffer.remaining() >= 8) {
            val chunkId = ByteArray(4)
            buffer.get(chunkId)
            val chunkSize = buffer.int
            val chunkName = String(chunkId)

            when (chunkName) {
                "fmt " -> {
                    if (chunkSize < 16) {
                        throw IllegalArgumentException("Corrupted WAV: 'fmt ' chunk size ($chunkSize) is invalid")
                    }
                    formatCode = buffer.short.toInt() and 0xFFFF
                    channels = buffer.short.toInt() and 0xFFFF
                    sampleRate = buffer.int
                    val byteRate = buffer.int
                    val blockAlign = buffer.short.toInt() and 0xFFFF
                    bitsPerSample = buffer.short.toInt() and 0xFFFF

                    // Skip any extra format bytes
                    val extraFmtBytes = chunkSize - 16
                    if (extraFmtBytes > 0) {
                        buffer.position(buffer.position() + extraFmtBytes)
                    }
                    fmtFound = true
                }
                "data" -> {
                    dataOffset = buffer.position()
                    dataSize = chunkSize
                    break
                }
                else -> {
                    // Skip unknown chunk
                    if (chunkSize > 0 && buffer.remaining() >= chunkSize) {
                        buffer.position(buffer.position() + chunkSize)
                    } else {
                        break
                    }
                }
            }
        }

        if (!fmtFound) {
            throw IllegalArgumentException("Malformed WAV file: Missing 'fmt ' chunk")
        }
        if (dataOffset == -1 || dataSize <= 0) {
            throw IllegalArgumentException("Malformed WAV file: Missing or empty 'data' chunk")
        }

        // 3. Strict Preprocessing Contract Checks
        if (channels != REQUIRED_CHANNELS) {
            throw IllegalArgumentException(
                "Unsupported channel count: expected mono ($REQUIRED_CHANNELS channel), but found $channels channels. " +
                "Stereo/multichannel audio is rejected without downmixing."
            )
        }

        if (sampleRate != REQUIRED_SAMPLE_RATE) {
            throw IllegalArgumentException(
                "Unsupported sample rate: expected $REQUIRED_SAMPLE_RATE Hz, but found $sampleRate Hz. " +
                "Silent resampling is strictly prohibited."
            )
        }

        val bytesPerSample = bitsPerSample / 8
        if (bytesPerSample <= 0) {
            throw IllegalArgumentException("Invalid bits per sample: $bitsPerSample")
        }

        val totalSamples = dataSize / bytesPerSample
        if (totalSamples != REQUIRED_SAMPLES) {
            throw IllegalArgumentException(
                "Unsupported duration: audio contains $totalSamples samples (${totalSamples.toFloat() / sampleRate}s). " +
                "Expected exactly $REQUIRED_SAMPLES samples (1.000s)."
            )
        }

        // 4. Sample Conversion to float32 in [-1.0, 1.0]
        val audioData = FloatArray(REQUIRED_SAMPLES)
        buffer.position(dataOffset)

        when (formatCode) {
            1 -> { // 16-bit PCM integer
                if (bitsPerSample != 16) {
                    throw IllegalArgumentException("Unsupported PCM bit depth: $bitsPerSample bits (expected 16-bit)")
                }
                for (i in 0 until REQUIRED_SAMPLES) {
                    val pcm16 = buffer.short
                    audioData[i] = pcm16.toFloat() / 32768.0f
                }
            }
            3 -> { // 32-bit IEEE float
                if (bitsPerSample != 32) {
                    throw IllegalArgumentException("Unsupported float bit depth: $bitsPerSample bits (expected 32-bit)")
                }
                for (i in 0 until REQUIRED_SAMPLES) {
                    audioData[i] = buffer.float
                }
            }
            else -> {
                throw IllegalArgumentException("Unsupported WAV format code: $formatCode (expected 1=PCM or 3=IEEE Float)")
            }
        }

        // 5. Finite Value Verification (NaN and Inf rejection)
        for (i in 0 until REQUIRED_SAMPLES) {
            val s = audioData[i]
            if (s.isNaN() || s.isInfinite()) {
                throw IllegalArgumentException("Corrupted audio data: sample at index $i is non-finite ($s)")
            }
        }

        return WavInfo(
            sampleRate = sampleRate,
            channels = channels,
            bitsPerSample = bitsPerSample,
            formatCode = formatCode,
            totalSamples = totalSamples,
            durationSec = totalSamples.toFloat() / sampleRate,
            audioData = audioData
        )
    }
}
