package com.example.synthetic_kws

import android.app.Activity
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import java.io.IOException

/**
 * Main Activity for Synthetic Vietnamese Keyword Spotting (KWS) Demo.
 * Demonstrates offline end-to-end KWS pipeline on pre-packaged synthetic fixtures:
 *   WAV Validation -> Native C++ Frontend (JNI) -> ONNX Runtime -> Game Action Adapter
 *
 * GOVERNANCE INVARIANT: HUMAN DATA COLLECTION REMAINS STRICTLY BLOCKED.
 * Purely offline execution over synthetic acoustic fixtures.
 */
class MainActivity : Activity() {

    private lateinit var spinnerFixtures: Spinner
    private lateinit var btnRunInference: Button
    private lateinit var txtValidationInfo: TextView
    private lateinit var txtResultLabel: TextView
    private lateinit var txtConfidence: TextView
    private lateinit var txtAction: TextView
    private lateinit var txtProbabilities: TextView
    private lateinit var txtTiming: TextView

    private var classifier: OnnxKwsClassifier? = null

    private val demoFixtures = arrayOf(
        "case_01_trai_demo.wav",
        "case_02_phai_demo.wav",
        "case_03_len_demo.wav",
        "case_04_xuong_demo.wav",
        "case_05_unknown_demo.wav",
        "case_06_silence_demo.wav",
        "case_09_silence_pure_zeros.wav",
        "case_10_low_amplitude_ambient.wav",
        "case_11_transient_impulse.wav",
        "case_12_holdout_trai_native.wav",
        "invalid_stereo.wav"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        spinnerFixtures = findViewById(R.id.spinnerFixtures)
        btnRunInference = findViewById(R.id.btnRunInference)
        txtValidationInfo = findViewById(R.id.txtValidationInfo)
        txtResultLabel = findViewById(R.id.txtResultLabel)
        txtConfidence = findViewById(R.id.txtConfidence)
        txtAction = findViewById(R.id.txtAction)
        txtProbabilities = findViewById(R.id.txtProbabilities)
        txtTiming = findViewById(R.id.txtTiming)

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, demoFixtures)
        spinnerFixtures.adapter = adapter

        // Initialize Native C++ Frontend
        val initStatus = KwsFrontendJni.init()
        if (initStatus != KwsFrontendJni.KWS_SUCCESS) {
            Toast.makeText(this, "Native frontend init failed: $initStatus", Toast.LENGTH_LONG).show()
        }

        // Initialize ONNX Runtime Model from Assets
        try {
            val modelStream = assets.open("models/arm_b_tcresnet14.onnx")
            classifier = OnnxKwsClassifier(modelStream)
        } catch (e: Exception) {
            txtValidationInfo.text = "Failed to load ONNX model: ${e.message}"
        }

        btnRunInference.setOnClickListener {
            val selectedFixture = spinnerFixtures.selectedItem as String
            runPipeline(selectedFixture)
        }
    }

    private fun runPipeline(fixtureName: String) {
        val assetPath = "audio/$fixtureName"
        try {
            // 1. WAV Loading & Validation
            val startTimePrep = System.nanoTime()
            val wavInfo = assets.open(assetPath).use { stream ->
                WavAudioLoader.loadAndValidate(stream)
            }
            txtValidationInfo.text = "Validated: ${wavInfo.sampleRate} Hz | ${wavInfo.channels} ch | ${wavInfo.totalSamples} samples (1.000s)"

            // 2. Native C++ Log-Mel Feature Extraction (JNI)
            val features = FloatArray(KwsFrontendJni.TOTAL_FEATURES)
            val frontendStatus = KwsFrontendJni.computeLogMel(
                wavInfo.audioData,
                KwsFrontendJni.INPUT_SAMPLES,
                features
            )

            if (frontendStatus != KwsFrontendJni.KWS_SUCCESS) {
                val errorMsg = KwsFrontendJni.getErrorString(frontendStatus)
                txtValidationInfo.text = "Frontend Error ($frontendStatus): $errorMsg"
                clearResults()
                return
            }

            val endTimePrep = System.nanoTime()
            val prepLatencyMs = (endTimePrep - startTimePrep) / 1_000_000.0f

            // 3. ONNX Runtime Neural Inference
            val model = classifier
            if (model == null) {
                txtValidationInfo.text = "Error: ONNX model not initialized."
                return
            }

            val classResult = model.classify(features)

            // 4. Game Action Safety Adaptation
            val adaptedAction = GameActionAdapter.adapt(
                predictedLabel = classResult.predictedLabel,
                confidence = classResult.confidence,
                threshold = GameActionAdapter.DEFAULT_CONFIDENCE_THRESHOLD
            )

            // 5. Update UI
            txtResultLabel.text = "Predicted: ${classResult.predictedLabel}"
            txtConfidence.text = "Confidence: ${String.format("%.1f", classResult.confidence * 100)}%"
            txtAction.text = "Adapted Action: ${adaptedAction.action.actionName}\n${adaptedAction.explanation}"

            val probBuilder = StringBuilder("Probabilities:\n")
            for (i in classResult.probabilities.indices) {
                val label = GameActionAdapter.CLASS_LABELS[i]
                val prob = classResult.probabilities[i]
                probBuilder.append(String.format("  %-8s : %5.1f%%\n", label, prob * 100))
            }
            txtProbabilities.text = probBuilder.toString()

            txtTiming.text = String.format("Timing: Preproc %.2f ms | Inference %.2f ms", prepLatencyMs, classResult.inferenceLatencyMs)

        } catch (e: IllegalArgumentException) {
            // Audio validation failure (e.g. stereo, wrong duration)
            txtValidationInfo.text = "REJECTED (Validation): ${e.message}"
            clearResults()
        } catch (e: Exception) {
            txtValidationInfo.text = "Error: ${e.message}"
            clearResults()
        }
    }

    private fun clearResults() {
        txtResultLabel.text = "Predicted: —"
        txtConfidence.text = "Confidence: —"
        txtAction.text = "Action: NO_ACTION"
        txtProbabilities.text = ""
        txtTiming.text = ""
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()
    }
}
