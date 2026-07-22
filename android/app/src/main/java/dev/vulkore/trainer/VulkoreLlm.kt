package dev.vulkore.trainer

/**
 * Thin binding over llm_jni.cpp — Gemma 3 1B running on the Adreno through Vulkore.
 *
 * THREADING: every method, `create` and `destroy` included, must be called from
 * the SAME thread. `vulkore::Context` is single-threaded and that includes
 * teardown. `ChatVM` owns a dedicated single-thread executor for exactly this.
 *
 * Streaming is deliberate: [step] returns one token so the UI can render text
 * as it arrives. A blocking generate() would produce the same words with none
 * of the effect — watching it type is the whole point.
 */
class VulkoreLlm(
    modelPath: String,
    matvecSpv: String,
    transformerSpv: String,
    samplingSpv: String,
    // 8192, matching decode.hpp: with sliding-window attention applied, capacity
    // costs memory (952 MiB resident) but not time -- gpu ms/token is flat from
    // 2048 to 65536. What costs time is DEPTH, and 8192 is the deepest context
    // still served above reading speed.
    maxSeq: Int = 8192,
) {
    private var handle: Long =
        create(modelPath, matvecSpv, transformerSpv, samplingSpv, maxSeq)

    /** False when the native session failed to load — the UI must say so. */
    val ok: Boolean get() = handle != 0L

    fun prefill(text: String): Int = if (ok) prefill(handle, text) else -1

    /** One token's text, or null when the turn is complete. */
    fun step(): String? = if (ok) step(handle) else null

    fun metrics(): Metrics {
        if (!ok) return Metrics()
        val v = metrics(handle)
        return Metrics(
            ms = v[0], tokensPerSec = v[1], dispatches = v[2].toInt(),
            submits = v[3].toInt(), position = v[4].toInt(),
            recordMs = v[5], gpuMs = v[6],
        )
    }

    fun deviceName(): String = if (ok) deviceName(handle) else "no device"
    fun weightMiB(): Double = if (ok) weightBytes(handle) / (1024.0 * 1024.0) else 0.0
    fun kvMiB(): Double = if (ok) kvBytes(handle) / (1024.0 * 1024.0) else 0.0

    /** KV positions used / available — the conversation dies when they meet. */
    fun position(): Int = if (ok) position(handle) else 0
    fun maxSeq(): Int = if (ok) maxSeq(handle) else 0

    fun reset() { if (ok) reset(handle) }

    fun close() { if (ok) { destroy(handle); handle = 0L } }

    data class Metrics(
        val ms: Double = 0.0,
        val tokensPerSec: Double = 0.0,
        val dispatches: Int = 0,
        val submits: Int = 0,
        val position: Int = 0,
        val recordMs: Double = 0.0,
        val gpuMs: Double = 0.0,
    )

    private external fun create(
        model: String, matvecSpv: String, transformerSpv: String,
        samplingSpv: String, maxSeq: Int,
    ): Long
    private external fun prefill(h: Long, text: String): Int
    private external fun step(h: Long): String?
    private external fun metrics(h: Long): DoubleArray
    private external fun deviceName(h: Long): String
    private external fun weightBytes(h: Long): Long
    private external fun kvBytes(h: Long): Long
    private external fun position(h: Long): Int
    private external fun maxSeq(h: Long): Int
    private external fun reset(h: Long)
    private external fun destroy(h: Long)

    companion object { init { System.loadLibrary("vulkore_llm") } }
}
