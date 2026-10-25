import Foundation
import Accelerate

/**
 * Mobile Interface Specification for Vietnamese Keyword Spotting (variant_w10).
 * Platform: iOS / macOS (Swift 5.9+).
 * Phase: 92 (Cross-Platform Audio Preprocessing Parity)
 * Status: NOT_BUILT (Blocked on host: Windows operating system, Xcode / macOS SDK unavailable).
 *
 * Architecture:
 *   - Model: CorrectedVietnameseKWSNet (138,469 parameters)
 *   - Runtime: ONNX Runtime Mobile for iOS or Apple CoreML
 *   - Frontend: vDSP FFT + Mel filterbank or C++ native kws_mobile_frontend_v10
 *   - Classes (5): TRAI (0), PHAI (1), LEN (2), XUONG (3), REJECT (4)
 *   - Threshold: theta_cmd = 0.40
 */
public final class VietnameseKwsClassifierV10 {

    public static let sampleRate: Int = 16000
    public static let bufferSamples: Int = 16000
    public static let melBands: Int = 40
    public static let timeFrames: Int = 101
    public static let totalFeatures: Int = 4040

    public static let classNames: [String] = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]
    public static let defaultThreshold: Float = 0.40

    public struct RecognitionResult {
        public let predictedLabel: String
        public let classIndex: Int
        public let rejectionStatus: String
        public let confidence: Float
        public let commandThreshold: Float
        public let probabilities: [Float]
        public let inferenceLatencyMs: Float
        public let preprocessingLatencyMs: Float
    }

    public let thetaCmd: Float

    public init(thetaCmd: Float = VietnameseKwsClassifierV10.defaultThreshold) {
        self.thetaCmd = thetaCmd
    }

    /**
     * MODE 1: Recorded-File Offline Inference.
     * Takes a URL to a 16kHz mono WAV file on the iOS filesystem.
     */
    public func classifyWavFile(at fileURL: URL) throws -> RecognitionResult {
        throw NSError(
            domain: "VietnameseKwsClassifierV10",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "NOT_BUILT: Xcode and iOS build environment unavailable on current host."]
        )
    }

    /**
     * MODE 2: Push-to-Talk (PTT) 1-Second Audio Buffer Inference.
     * Takes exactly 16,000 float32 PCM samples captured via AVAudioEngine.
     */
    public func classifyPttBuffer(_ pcmBuffer: [Float]) throws -> RecognitionResult {
        guard pcmBuffer.count == Self.bufferSamples else {
            throw NSError(
                domain: "VietnameseKwsClassifierV10",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Expected exactly \(Self.bufferSamples) samples, got \(pcmBuffer.count)"]
            )
        }
        throw NSError(
            domain: "VietnameseKwsClassifierV10",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "NOT_BUILT: Xcode and iOS build environment unavailable on current host."]
        )
    }

    /**
     * Decision and Threshold Gating Logic.
     */
    public func evaluateThresholdGating(
        probabilities: [Float],
        prepLatencyMs: Float,
        infLatencyMs: Float
    ) -> RecognitionResult {
        precondition(probabilities.count == 5, "Expected 5 class probabilities")

        var maxCmdIdx = 0
        var maxCmdProb = probabilities[0]
        for i in 1..<4 {
            if probabilities[i] > maxCmdProb {
                maxCmdProb = probabilities[i]
                maxCmdIdx = i
            }
        }

        let isAccepted = maxCmdProb >= thetaCmd
        let label = isAccepted ? Self.classNames[maxCmdIdx] : Self.classNames[4]
        let status = isAccepted ? "ACCEPTED" : "REJECTED"
        let confidence = isAccepted ? maxCmdProb : probabilities[4]
        let classIdx = isAccepted ? maxCmdIdx : 4

        return RecognitionResult(
            predictedLabel = label,
            classIndex = classIdx,
            rejectionStatus = status,
            confidence: confidence,
            commandThreshold: thetaCmd,
            probabilities: probabilities,
            inferenceLatencyMs: infLatencyMs,
            preprocessingLatencyMs: prepLatencyMs
        )
    }
}
