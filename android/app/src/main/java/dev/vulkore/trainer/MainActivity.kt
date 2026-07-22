package dev.vulkore.trainer

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.systemBars
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.viewmodel.compose.viewModel

private val Ink = Color(0xFF0B0F14)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme(colorScheme = darkColorScheme()) { Root() } }
    }
}

@Composable
private fun Root() {
    // CHAT only. ChatVM is the app's single ViewModel and it does not touch the
    // GPU in its init block, so launching the app touches the GPU exactly once:
    // when you press LOAD and Gemma's weights go onto the device.
    val chatVm: ChatVM = viewModel()

    Column(Modifier.fillMaxSize().background(Ink)
        .windowInsetsPadding(WindowInsets.systemBars)) {
        ChatScreen(chatVm)
    }
}
