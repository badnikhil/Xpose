// JNI shim over vulkore::llm::DecodeSession — Gemma 3 1B on the phone GPU.
//
// THREADING: every entry point, create and
// destroy included, must run on the ONE thread that owns the session, because
// vulkore::Context is single-threaded and that includes teardown. Kotlin enforces
// it with a dedicated executor.
//
// STREAMING: the UI needs a token at a time, so this deliberately exposes
// `step()` returning one piece of text rather than a blocking `generate()`.
// A blocking call cannot animate, and watching the text appear IS the demo.

#include <jni.h>
#include <android/log.h>
#include <string>
#include <vector>

#include "decode.hpp"
#include "tokenizer.hpp"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "VulkoreLlm", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "VulkoreLlm", __VA_ARGS__)

namespace {

// Owns the session plus the small amount of per-turn state the UI needs. Kept
// here rather than in DecodeSession because it is chat framing, not inference.
struct ChatSession {
    vulkore::llm::DecodeSession sess;
    uint32_t                  eos = 0;
    uint32_t                  eot = 0;
    bool                      done = false;

    ChatSession(const vulkore::llm::Paths& p, const vulkore::llm::DecodeConfig& c)
        : sess(p, c) {
        // Gemma ends a turn with <end_of_turn>, not </s>; stopping only on EOS
        // makes the model run on into a fabricated user turn.
        const auto& tk = sess.tokenizer();
        auto ids = tk.encode("<end_of_turn>", false);
        eot = ids.empty() ? 0u : uint32_t(ids.back());
        eos = tk.eos_id() < 0 ? 0u : uint32_t(tk.eos_id());
    }
};

ChatSession* as_chat(jlong h) { return reinterpret_cast<ChatSession*>(h); }

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out(c ? c : "");
    env->ReleaseStringUTFChars(s, c);
    return out;
}

}  // namespace

extern "C" {

/** Loads weights and kernels. Slow (~7 s) — never call from the UI thread. */
JNIEXPORT jlong JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_create(JNIEnv* env, jobject, jstring model,
                                       jstring matvecSpv, jstring transformerSpv,
                                       jstring samplingSpv, jint maxSeq) {
    vulkore::llm::Paths paths;
    paths.model_gguf       = jstr(env, model);
    paths.matvec.module    = jstr(env, matvecSpv);
    paths.transformer_spv  = jstr(env, transformerSpv);
    paths.sampling_spv     = jstr(env, samplingSpv);

    vulkore::llm::DecodeConfig cfg;
    cfg.max_seq = uint32_t(maxSeq);

    // Greedy argmax is the default and it is WRONG for a chat UI: a 1B model
    // decoded greedily falls into repetition loops ("to to to to ..."), and our
    // weights make that worse — they are double-quantised, correlating 0.95
    // with fp32 rather than 0.999, so the top-1 logit is often a near-tie that
    // greedy resolves the same way every step.
    //
    // Temperature + top-k is Gemma's own recommended decoding. It costs a
    // full-vocab logit download per token instead of two floats, which is a
    // real throughput hit — but a fast wrong answer is not a demo.
    // Gemma's own recommended decoding, now that top-p exists. Temperature is
    // held at 0.7 rather than 1.0: this is a 1B model whose weights are
    // double-quantised, and it does not have the knowledge to be creative with.
    // top-p is what actually suppresses confident-sounding nonsense -- it drops
    // the low-probability tail that top-k alone keeps in play.
    cfg.sampling.mode        = vulkore::llm::SampleMode::Temperature;
    cfg.sampling.temperature = 0.7f;
    cfg.sampling.top_k       = 64;
    cfg.sampling.top_p       = 0.95f;
    cfg.sampling.seed        = 1234;

    try {
        auto* c = new ChatSession(paths, cfg);
        LOGI("llm ready on %s | weights %.1f MiB | kv %.1f MiB",
             c->sess.device_name().c_str(),
             double(c->sess.weight_bytes()) / (1024.0 * 1024.0),
             double(c->sess.kv_bytes()) / (1024.0 * 1024.0));
        return reinterpret_cast<jlong>(c);
    } catch (const std::exception& ex) {
        LOGE("llm create failed: %s", ex.what());
        return 0;
    }
}

/**
 * Wraps `text` in Gemma's chat template, tokenises, and runs prefill. Returns
 * prompt tokens processed, or -1 on failure.
 */
JNIEXPORT jint JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_prefill(JNIEnv* env, jobject, jlong h, jstring text) {
    auto* c = as_chat(h);
    if (!c) return -1;
    try {
        // A SYSTEM INSTRUCTION, prepended to the first user turn (Gemma 3 has no
        // separate system role). Without one the model treats every input as a
        // request to elaborate: "cool" produced three more paragraphs of story,
        // and "no need to do that" was followed by unrelated code. A 1B model
        // will not infer brevity on its own; it has to be told, every turn.
        // NO SYSTEM PROMPT. Two were tried and both made things worse: the
        // first ("as few words as possible") made it answer "Okay." to a direct
        // question, the second still classified "hey" as an acknowledgement and
        // replied "Okay." again. Both existed to paper over the repeated-BOS bug
        // below, which was the actual cause of the behaviour they were aimed at.
        // Gemma 3 1B-*it* is already instruction-tuned for chat; it does not need
        // to be told how to hold a conversation, and ~90 tokens of instruction
        // was buying nothing but a worse first turn.
        const bool first = c->sess.position() == 0;

        // Gemma's template closes the model's turn with "<end_of_turn>\n". The
        // model samples <end_of_turn> itself (and it IS in the KV cache), but the
        // trailing newline is ours to emit, so later turns must open with it.
        std::string prompt = std::string(first ? "" : "\n") +
                             "<start_of_turn>user\n" +
                             jstr(env, text) +
                             "<end_of_turn>\n<start_of_turn>model\n";

        // BOS ONLY ONCE, on the first turn. This was hardcoded true, so every
        // turn injected a document-start token into the middle of the
        // conversation -- the strongest possible signal that everything before it
        // has ended. The model duly answered as if freshly started with no
        // context, which is why "hey" came back as "Okay." and why multi-turn
        // replies drifted onto unrelated topics. Not a sampling or prompt
        // problem; the conversation itself was malformed from turn two onward.
        auto ids = c->sess.tokenizer().encode(prompt, /*add_bos=*/first);
        std::vector<uint32_t> u(ids.begin(), ids.end());
        c->sess.prefill(u);
        c->done = false;
        return jint(u.size());
    } catch (const std::exception& ex) {
        LOGE("prefill failed: %s", ex.what());
        return -1;
    }
}

/**
 * Generates ONE token and returns its text, or null once the turn is over.
 * Null is the stop signal — the caller loops until it sees one.
 */
JNIEXPORT jstring JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_step(JNIEnv* env, jobject, jlong h) {
    auto* c = as_chat(h);
    if (!c || c->done) return nullptr;
    try {
        uint32_t tok = c->sess.decode_step();
        if (tok == c->eot || tok == c->eos) { c->done = true; return nullptr; }
        // Out of KV cache: stop cleanly rather than corrupting the next turn.
        if (c->sess.position() + 1 >= c->sess.config().max_seq) c->done = true;
        return env->NewStringUTF(c->sess.tokenizer().decode_one(int32_t(tok)).c_str());
    } catch (const std::exception& ex) {
        LOGE("step failed: %s", ex.what());
        c->done = true;
        return nullptr;
    }
}

/** {ms, tok/s, dispatches, submits, position, record_ms, gpu_ms} for the last step. */
JNIEXPORT jdoubleArray JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_metrics(JNIEnv* env, jobject, jlong h) {
    jdoubleArray out = env->NewDoubleArray(7);
    if (!h) return out;
    auto m = as_chat(h)->sess.last_metrics();
    double v[7] = { m.ms, m.tokens_per_sec, double(m.dispatches), double(m.submits),
                    double(m.position), m.record_ms, m.gpu_ms };
    env->SetDoubleArrayRegion(out, 0, 7, v);
    return out;
}

JNIEXPORT jstring JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_deviceName(JNIEnv* env, jobject, jlong h) {
    return env->NewStringUTF(h ? as_chat(h)->sess.device_name().c_str() : "no device");
}

/** Weight bytes resident on the GPU — the honest "this is really here" number. */
JNIEXPORT jlong JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_weightBytes(JNIEnv*, jobject, jlong h) {
    return h ? jlong(as_chat(h)->sess.weight_bytes()) : 0;
}

JNIEXPORT jlong JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_kvBytes(JNIEnv*, jobject, jlong h) {
    return h ? jlong(as_chat(h)->sess.kv_bytes()) : 0;
}

/** KV positions used, so the UI can warn before the context is exhausted. */
JNIEXPORT jint JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_position(JNIEnv*, jobject, jlong h) {
    return h ? jint(as_chat(h)->sess.position()) : 0;
}

JNIEXPORT jint JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_maxSeq(JNIEnv*, jobject, jlong h) {
    return h ? jint(as_chat(h)->sess.config().max_seq) : 0;
}

/** Clears the KV cache. Weights stay resident, so a new chat is instant. */
JNIEXPORT void JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_reset(JNIEnv*, jobject, jlong h) {
    if (h) { as_chat(h)->sess.reset(); as_chat(h)->done = false; }
}

JNIEXPORT void JNICALL
Java_dev_vulkore_trainer_VulkoreLlm_destroy(JNIEnv*, jobject, jlong h) {
    delete as_chat(h);   // must be the owning thread
}

}  // extern "C"
