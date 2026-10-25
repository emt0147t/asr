package com.example.synthetic_kws

/**
 * Game Control Adapter for Android Synthetic Vietnamese KWS.
 * Decouples raw classification probabilities from game action dispatch:
 *   - TRAI    -> MOVE_LEFT
 *   - PHAI    -> MOVE_RIGHT
 *   - LEN     -> JUMP
 *   - XUONG   -> CROUCH
 *   - UNKNOWN -> NO_ACTION
 *   - SILENCE -> NO_ACTION
 *
 * Enforces strict safety suppression:
 * 1. Non-target classes unconditionally yield NO_ACTION.
 * 2. Target commands with confidence below threshold yield NO_ACTION.
 */
object GameActionAdapter {

    enum class GameAction(val actionName: String, val dx: Int, val dy: Int) {
        MOVE_LEFT("MOVE_LEFT", -1, 0),
        MOVE_RIGHT("MOVE_RIGHT", 1, 0),
        JUMP("JUMP", 0, -1),
        CROUCH("CROUCH", 0, 1),
        NO_ACTION("NO_ACTION", 0, 0)
    }

    data class AdaptedAction(
        val action: GameAction,
        val sourceLabel: String,
        val confidence: Float,
        val threshold: Float,
        val isActive: Boolean,
        val explanation: String
    )

    const val DEFAULT_CONFIDENCE_THRESHOLD = 0.50f

    val CLASS_LABELS = arrayOf("TRAI", "PHAI", "LEN", "XUONG", "UNKNOWN", "SILENCE")

    fun adapt(
        predictedLabel: String,
        confidence: Float,
        threshold: Float = DEFAULT_CONFIDENCE_THRESHOLD
    ): AdaptedAction {
        // 1. Non-target safety suppression: UNKNOWN and SILENCE never trigger character motion
        if (predictedLabel == "UNKNOWN" || predictedLabel == "SILENCE") {
            return AdaptedAction(
                action = GameAction.NO_ACTION,
                sourceLabel = predictedLabel,
                confidence = confidence,
                threshold = threshold,
                isActive = false,
                explanation = "Non-target class ($predictedLabel) safely suppressed to NO_ACTION."
            )
        }

        // 2. Experimental confidence threshold suppression
        if (confidence < threshold) {
            return AdaptedAction(
                action = GameAction.NO_ACTION,
                sourceLabel = predictedLabel,
                confidence = confidence,
                threshold = threshold,
                isActive = false,
                explanation = "Confidence ($confidence) is below safety threshold ($threshold); suppressed to NO_ACTION."
            )
        }

        // 3. Valid target command dispatch
        val action = when (predictedLabel) {
            "TRAI" -> GameAction.MOVE_LEFT
            "PHAI" -> GameAction.MOVE_RIGHT
            "LEN" -> GameAction.JUMP
            "XUONG" -> GameAction.CROUCH
            else -> GameAction.NO_ACTION
        }

        return AdaptedAction(
            action = action,
            sourceLabel = predictedLabel,
            confidence = confidence,
            threshold = threshold,
            isActive = (action != GameAction.NO_ACTION),
            explanation = "Command '$predictedLabel' accepted with ${(confidence * 100).toInt()}% confidence -> ${action.actionName}"
        )
    }
}
