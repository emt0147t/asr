package com.example.synthetic_kws

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.InputStream
import java.nio.FloatBuffer
import java.util.Collections

/**
 * Android ONNX Runtime wrapper for Vietnamese TC-ResNet-14 KWS model.
 * Executes offline neural inference from 40x101 Log-Mel features.
 */
class OnnxKwsClassifier(modelInputStream: InputStream) : AutoCloseable {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    data class ClassificationResult(
        val predictedLabel: String,
        val predictedClassId: Int,
        val confidence: Float,
        val probabilities: FloatArray,
        val logits: FloatArray,
        val inferenceLatencyMs: Float
    )

    init {
        val modelBytes = modelInputStream.readBytes()
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
        }
        session = env.createSession(modelBytes, sessionOptions)
    }

    fun classify(features: FloatArray): ClassificationResult {
        require(features.size == KwsFrontendJni.TOTAL_FEATURES) {
            "Expected ${KwsFrontendJni.TOTAL_FEATURES} feature elements, got ${features.size}"
        }

        val startTime = System.nanoTime()

        // Shape: (Batch=1, Channels=1, Mel=40, Time=101)
        val shape = longArrayOf(1, 1, KwsFrontendJni.MEL_BANDS.toLong(), KwsFrontendJni.TIME_FRAMES.toLong())
        val floatBuffer = FloatBuffer.wrap(features)
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, shape)

        val results = session.run(Collections.singletonMap("log_mel", inputTensor))
        val endTime = System.nanoTime()
        val latencyMs = (endTime - startTime) / 1_000_000.0f

        // Model outputs: 'logits' (1, 6) and 'probabilities' (1, 6)
        val rawLogits = (results.get(0).value as Array<FloatArray>)[0]
        val rawProbs = (results.get(1).value as Array<FloatArray>)[0]

        inputTensor.close()
        results.close()

        // Find argmax class
        var maxIdx = 0
        var maxProb = rawProbs[0]
        for (i in 1 until rawProbs.size) {
            if (rawProbs[i] > maxProb) {
                maxProb = rawProbs[i]
                maxIdx = i
            }
        }

        val label = if (maxIdx in GameActionAdapter.CLASS_LABELS.indices) {
            GameActionAdapter.CLASS_LABELS[maxIdx]
        } else {
            "UNKNOWN"
        }

        return ClassificationResult(
            predictedLabel = label,
            predictedClassId = maxIdx,
            confidence = maxProb,
            probabilities = rawProbs,
            logits = rawLogits,
            inferenceLatencyMs = latencyMs
        )
    }

    override fun close() {
        session.close()
        env.close()
    }
}
