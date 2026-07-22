package dev.vulkore.trainer

import android.app.Application
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors

private val Ink = Color(0xFF0B0F14)
private val Panel = Color(0xFF141B24)
private val Cyan = Color(0xFF22D3EE)
private val Dim = Color(0xFF64748B)
private val Text_ = Color(0xFFE2E8F0)
private val Warn = Color(0xFFF59E0B)

/**
 * The model file is ~762 MB, far too large for an APK. It is side-loaded to the
 * app's external files dir, which needs no runtime permission:
 *
 *   adb push gemma-3-1b-it-Q4_K_M.gguf \
 *     /sdcard/Android/data/dev.vulkore.trainer/files/
 */
private const val MODEL_FILE = "gemma-3-1b-it-Q4_K_M.gguf"

data class ChatMsg(val user: Boolean, val text: String)

class ChatVM(app: Application) : AndroidViewModel(app) {

    // ONE thread owns the native session for its entire lifetime, creation and
    // teardown included — vulkore::Context is single-threaded, teardown included.
    private val exec = Executors.newSingleThreadExecutor { r -> Thread(r, "vulkore-llm") }
    private val dispatcher = exec.asCoroutineDispatcher()

    private var llm: VulkoreLlm? = null

    // Set from the UI thread, read by the token loop on the owning thread. The
    // loop only ever reads it, so @Volatile is enough -- no lock, and nothing
    // that could block the generation thread.
    @Volatile private var stopRequested = false

    var messages by mutableStateOf(listOf<ChatMsg>()); private set
    var input by mutableStateOf(""); private set
    var status by mutableStateOf("tap LOAD to bring Gemma onto the GPU"); private set
    var device by mutableStateOf(""); private set
    var busy by mutableStateOf(false); private set
    var loaded by mutableStateOf(false); private set
    var tokPerSec by mutableStateOf(0.0); private set
    var dispatches by mutableStateOf(0); private set
    var residentMiB by mutableStateOf(0.0); private set
    var ctxUsed by mutableStateOf(0); private set
    var ctxMax by mutableStateOf(0); private set

    fun onInput(s: String) { input = s }

    /**
     * Interrupt generation. The partial reply STAYS in the transcript and stays
     * in the KV cache, because the tokens really were generated -- discarding
     * them on screen while the model still remembers them would desynchronise
     * the visible conversation from the one the model is conditioned on.
     */
    fun stop() { stopRequested = true }

    /** Loading takes ~7 s (weight repack), so it is explicit rather than implicit. */
    fun load() {
        if (busy || loaded) return
        busy = true
        viewModelScope.launch {
            val ctx = getApplication<Application>().applicationContext
            val model = File(ctx.getExternalFilesDir(null), MODEL_FILE)
            if (!model.exists()) {
                status = "model not found — push $MODEL_FILE to ${model.parent}"
                busy = false
                return@launch
            }
            status = "loading weights onto the GPU…"
            val ok = withContext(dispatcher) {
                // ALWAYS refresh the kernels from assets: copying only when
                // absent left stale .spv in filesDir across upgrades, so new
                // kernels silently never ran.
                val spv = listOf("llm_matvec.spv", "llm_transformer.spv", "llm_sampling.spv")
                    .map { name ->
                        File(ctx.filesDir, name).also { f ->
                            ctx.assets.open(name).use { i -> f.outputStream().use { i.copyTo(it) } }
                        }.absolutePath
                    }
                val s = VulkoreLlm(model.absolutePath, spv[0], spv[1], spv[2])
                if (s.ok) { llm = s; true } else { s.close(); false }
            }
            if (ok) {
                val s = llm!!
                device = s.deviceName()
                residentMiB = s.weightMiB() + s.kvMiB()
                ctxMax = s.maxSeq()
                loaded = true
                status = "ready — offline, on the GPU"
            } else {
                status = "GPU session failed — see logcat VulkoreLlm"
            }
            busy = false
        }
    }

    fun send() {
        val prompt = input.trim()
        val s = llm ?: return
        if (prompt.isEmpty() || busy || !loaded) return

        // The KV cache is finite (2048 positions). Once it fills, decode_step()
        // returns null immediately and the chat dies SILENTLY with no error --
        // which is exactly what "I can't chat any more" looked like. Reset and
        // say so, rather than accepting a message that can never be answered.
        // Reserve room for the prompt AND a full reply. A generous estimate of
        // 1 token per 3 chars, plus 256 for the answer; being early is free,
        // being late loses the turn.
        val need = prompt.length / 3 + 256
        if (ctxMax > 0 && ctxUsed + need > ctxMax) {
            viewModelScope.launch {
                withContext(dispatcher) { llm?.reset() }
                ctxUsed = 0
                messages = messages + ChatMsg(
                    false, "_(context full — history cleared, weights stayed on the GPU)_")
            }
            return
        }
        input = ""
        messages = messages + ChatMsg(true, prompt) + ChatMsg(false, "")
        busy = true
        status = "thinking…"

        // The ENTIRE token loop runs on the owning thread. The earlier version
        // did `withContext(dispatcher)` twice per token — one hop for step(),
        // another for metrics() — so every token paid two thread handoffs plus
        // a JNI round trip it did not need. That alone was most of the gap
        // between the app (16 tok/s) and the same session driven from the CLI
        // (32 tok/s): the model was never the bottleneck, my plumbing was.
        viewModelScope.launch(dispatcher) {
            // prefill returns -1 when it throws -- overwhelmingly because the KV
            // cache cannot fit the prompt. Ignoring it was fatal: the loop then
            // ran, step() returned null on the first call, and the user got an
            // empty reply with no error and a context counter still reading low.
            val nPrompt = s.prefill(prompt)
            if (nPrompt < 0) {
                val used = s.position()
                viewModelScope.launch {
                    ctxUsed = used
                    messages = messages.dropLast(1) + ChatMsg(
                        false, "_(out of context at $used/${s.maxSeq()} — tap NEW to start over)_")
                    status = "context full"
                    busy = false
                }
                return@launch
            }
            stopRequested = false
            val sb = StringBuilder()
            var n = 0
            val t0 = System.nanoTime()
            var lastPost = 0L
            var disp = 0
            var pos = 0

            fun post(done: Boolean) {
                val text = sb.toString()
                // Read the KV position on EVERY post, including the final one.
                // Sampling it only inside the 33 ms repaint tick meant a short
                // reply never updated it, so the counter under-reported for the
                // whole session and the guard below never fired.
                pos = s.position()
                val secs = (System.nanoTime() - t0) / 1e9
                // AVERAGE over this turn, not the instantaneous last-step rate.
                // Per-token throughput decays as the KV cache fills (attention
                // cost grows with position), so the instantaneous figure sags
                // visibly mid-reply. The average is steadier and is the number
                // comparable to LiteRT-LM's reported decode speed.
                val tps = if (secs > 0) n / secs else 0.0
                viewModelScope.launch {
                    tokPerSec = tps
                    dispatches = disp
                    ctxUsed = pos
                    messages = messages.dropLast(1) + ChatMsg(false, text)
                    if (done) { status = "ready — offline, on the GPU"; busy = false }
                }
            }

            while (true) {
                if (stopRequested) break
                val piece = s.step() ?: break
                sb.append(piece)
                n++
                // Repaint at ~30 fps, NOT once per token. Every repaint re-parses
                // the whole accumulated reply for markdown, so posting per token
                // made rendering O(n^2) in reply length — the thing that made
                // long answers crawl. Tokens still stream; the UI just coalesces.
                val now = System.nanoTime()
                if (now - lastPost > 33_000_000L) {
                    lastPost = now
                    disp = s.metrics().dispatches   // one JNI call per repaint
                    post(false)
                }
            }
            post(true)
        }
    }

    fun newChat() {
        if (busy) return
        viewModelScope.launch {
            withContext(dispatcher) { llm?.reset() }
            ctxUsed = 0
            messages = listOf()
        }
    }

    override fun onCleared() {
        // Destruction must also happen on the owning thread.
        exec.execute { llm?.close(); llm = null }
        exec.shutdown()
    }
}

@Composable
fun ChatScreen(vm: ChatVM) {
    val listState = rememberLazyListState()

    // Is the user parked at the bottom? If they have scrolled up to re-read
    // something, we must NOT drag them back down on every token.
    val atBottom by remember {
        derivedStateOf {
            val last = listState.layoutInfo.visibleItemsInfo.lastOrNull()
                ?: return@derivedStateOf true
            last.index >= listState.layoutInfo.totalItemsCount - 1 &&
                last.offset + last.size <= listState.layoutInfo.viewportEndOffset + 96
        }
    }

    // Follow the stream only while they are at the bottom, and scroll INSTANTLY
    // rather than animating: a token arrives every ~22 ms, so each animation was
    // being cancelled by the next one before it finished. That interrupted
    // animation is what read as broken scrolling.
    LaunchedEffect(vm.messages.size, vm.messages.lastOrNull()?.text) {
        if (vm.messages.isNotEmpty() && atBottom) {
            val i = vm.messages.size - 1
            listState.scrollToItem(i, Int.MAX_VALUE / 2)   // pin to item's end
        }
    }

    Column(Modifier.fillMaxSize().background(Ink).padding(12.dp)) {

        // ---- header: what is actually running, stated plainly ----------------
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("GEMMA 3 1B · ON-DEVICE", color = Text_, fontSize = 15.sp,
                     fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                Text(vm.status, color = if (vm.loaded) Dim else Warn, fontSize = 11.sp,
                     fontFamily = FontFamily.Monospace)
            }
            if (!vm.loaded) {
                Button(onClick = { vm.load() }, enabled = !vm.busy,
                       colors = ButtonDefaults.buttonColors(containerColor = Cyan)) {
                    Text(if (vm.busy) "LOADING" else "LOAD", color = Ink, fontSize = 12.sp,
                         fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                }
            } else {
                // Two taps: NEW wipes the conversation and a stray press on a
                // phone-sized target lost one.
                var confirmNew by remember { mutableStateOf(false) }
                Button(onClick = { if (confirmNew) { vm.newChat(); confirmNew = false }
                                   else confirmNew = true },
                       enabled = !vm.busy,
                       colors = ButtonDefaults.buttonColors(containerColor = Panel)) {
                    Text(if (confirmNew) "SURE?" else "NEW",
                         color = if (confirmNew) Warn else Cyan, fontSize = 11.sp,
                         fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                }
            }
        }

        Spacer(Modifier.height(10.dp))

        // ---- transcript ------------------------------------------------------
        LazyColumn(Modifier.weight(1f).fillMaxWidth(), state = listState,
                   verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(vm.messages) { m ->
                Column(Modifier.fillMaxWidth()) {
                    Text(if (m.user) "you" else "gemma",
                         color = if (m.user) Dim else Cyan, fontSize = 10.sp,
                         fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                    // A caret while the token loop is live: the difference
                    // between "it is generating" and "it has frozen".
                    val live = !m.user && vm.busy && m === vm.messages.last()
                    Text(
                        markdown(m.text + if (live) "▌" else ""),
                        color = Text_, fontSize = 14.sp, lineHeight = 20.sp,
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                            .background(if (m.user) Panel else Color(0xFF10202B))
                            .padding(10.dp),
                    )
                }
            }
        }

        // ---- live metrics: the claim, measured, on screen ---------------------
        if (vm.loaded) {
            // TWO rows, not one. Five stats in a single SpaceBetween row overflowed
            // on a phone width, and the failure mode was silent: the gaps collapsed
            // to zero and it rendered "Adreno (TM) 84032.2 tok/s836 disp".
            Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp))
                       .background(Panel).padding(horizontal = 10.dp, vertical = 7.dp),
                   verticalArrangement = Arrangement.spacedBy(3.dp)) {
                Row(Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(vm.device.removePrefix("Adreno (TM) ").let { "Adreno $it" },
                         color = Cyan, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                    Text("%.1f tok/s".format(vm.tokPerSec), color = Text_, fontSize = 10.sp,
                         fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                }
                Row(Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("${vm.dispatches} disp · 1 submit", color = Dim, fontSize = 10.sp,
                         fontFamily = FontFamily.Monospace)
                    Text("%.0f MiB".format(vm.residentMiB), color = Dim, fontSize = 10.sp,
                         fontFamily = FontFamily.Monospace)
                    if (vm.ctxMax > 0) {
                        val tight = vm.ctxUsed > vm.ctxMax - 512
                        Text("ctx ${vm.ctxUsed}/${vm.ctxMax}",
                             color = if (tight) Warn else Dim, fontSize = 10.sp,
                             fontFamily = FontFamily.Monospace)
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        // ---- composer --------------------------------------------------------
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = vm.input, onValueChange = vm::onInput,
                modifier = Modifier.weight(1f), enabled = vm.loaded && !vm.busy,
                placeholder = { Text("ask it anything", color = Dim, fontSize = 13.sp) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { vm.send() }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedTextColor = Text_, unfocusedTextColor = Text_,
                    focusedBorderColor = Cyan, unfocusedBorderColor = Dim,
                ),
            )
            // One button, two states. A separate STOP would sit dead most of the
            // time, and while generating SEND is unusable anyway.
            val generating = vm.busy && vm.loaded
            Button(onClick = { if (generating) vm.stop() else vm.send() },
                   enabled = vm.loaded,
                   colors = ButtonDefaults.buttonColors(
                       containerColor = if (generating) Warn else Cyan)) {
                Text(if (generating) "STOP" else "SEND", color = Ink, fontSize = 12.sp,
                     fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/**
 * Minimal markdown for chat output. Gemma emits `**bold**`, `*` bullets and
 * ``` fences constantly, and rendering them raw is the single thing that makes
 * on-device output look broken next to a hosted assistant.
 *
 * Deliberately NOT a markdown library: this runs per token during streaming, so
 * it must be cheap and must never throw on the half-finished syntax that a
 * partial stream always contains (an unclosed `**`, a fence with no end).
 */
private fun markdown(src: String): AnnotatedString = buildAnnotatedString {
    var i = 0
    var bold = false
    var code = false
    while (i < src.length) {
        val rest = src.length - i
        when {
            // ``` fence: flip monospace, swallow the marker and any language tag
            rest >= 3 && src.startsWith("```", i) -> {
                code = !code
                i += 3
                if (code) while (i < src.length && src[i] != '\n') i++   // drop "kotlin"
            }
            // **bold**
            rest >= 2 && src.startsWith("**", i) -> { bold = !bold; i += 2 }
            // "* " or "- " at line start becomes a real bullet
            (i == 0 || src[i - 1] == '\n') && rest >= 2 &&
                (src.startsWith("* ", i) || src.startsWith("- ", i)) -> {
                withStyle(SpanStyle(color = Cyan)) { append("  •  ") }
                i += 2
            }
            else -> {
                // Consume a RUN of plain text, not a single Char. Appending
                // char-by-char split surrogate pairs, so every emoji rendered as
                // two replacement diamonds — Gemma uses them constantly.
                // Kotlin Strings are UTF-16, so an astral code point is two Chars
                // that must stay together.
                var j = i
                while (j < src.length) {
                    val r = src.length - j
                    if (r >= 2 && src.startsWith("**", j)) break
                    if (r >= 3 && src.startsWith("```", j)) break
                    if (src[j] == '\n' && r >= 3 &&
                        (src.startsWith("* ", j + 1) || src.startsWith("- ", j + 1))) {
                        j++   // keep the newline, let the bullet rule take the rest
                        break
                    }
                    j++
                }
                if (j == i) j = i + 1          // never stall
                withStyle(
                    SpanStyle(
                        fontWeight = if (bold) FontWeight.Bold else FontWeight.Normal,
                        fontFamily = if (code) FontFamily.Monospace else FontFamily.Default,
                        color = if (code) Cyan else Text_,
                    )
                ) { append(src, i, j) }
                i = j
            }
        }
    }
}
