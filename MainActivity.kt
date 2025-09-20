package com.example.strawberrypickerapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.enableEdgeToEdge
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.strawberrypickerapp.ui.theme.StrawberryPickerAppTheme
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import androidx.camera.core.ImageProxy
import android.util.Log

class MainActivity : ComponentActivity() {

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val frameQueue: LinkedBlockingQueue<Bitmap> = LinkedBlockingQueue(5)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val classifier = Classifier(assets)
        var resultText by mutableStateOf("Waiting...")

        // ---- Option A: Auto classification every second ----
        Executors.newSingleThreadExecutor().execute {
            while (!isFinishing) {
                try {
                    val bitmap = frameQueue.take()
                    val result = classifier.classify(bitmap)
                    runOnUiThread { resultText = result }
                    Thread.sleep(1000) // classify 1 frame per second
                } catch (e: Exception) {
                    e.printStackTrace()
                    runOnUiThread { resultText = "⚠️ Error: ${e.message}" }
                }
            }
        }

        // Permission launcher
        val cameraPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (!granted) resultText = "Camera permission denied"
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            StrawberryPickerAppTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding)
                    ) {
                        // Camera preview
                        AndroidView(
                            factory = { ctx ->
                                PreviewView(ctx).apply {
                                    if (ContextCompat.checkSelfPermission(
                                            ctx,
                                            Manifest.permission.CAMERA
                                        ) == PackageManager.PERMISSION_GRANTED
                                    ) {
                                        startCamera(this) { bitmap ->
                                            frameQueue.offer(bitmap) // add to queue
                                        }
                                    }
                                }
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(600.dp)
                        )

                        Spacer(modifier = Modifier.height(16.dp))

                        // Result text
                        Text(
                            text = resultText,
                            style = MaterialTheme.typography.headlineMedium,
                            modifier = Modifier.padding(16.dp)
                        )

                        // ---- Option B: Manual capture button ----
                        Button(
                            onClick = {
                                val bitmap = frameQueue.poll()
                                if (bitmap != null) {
                                    val result = classifier.classify(bitmap)
                                    resultText = result
                                } else {
                                    resultText = "⚠️ No frame available"
                                }
                            },
                            modifier = Modifier.padding(16.dp)
                        ) {
                            Text("📸 Capture & Classify")
                        }
                    }
                }
            }
        }
    }

    private fun startCamera(
        previewView: PreviewView,
        onFrameCaptured: (Bitmap) -> Unit
    ) {
        val context = this
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = androidx.camera.core.Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                val bitmap = imageProxy.toBitmap()
                if (bitmap != null) {
                    Log.d("CameraX", "Frame captured: ${bitmap.width}x${bitmap.height}")
                    onFrameCaptured(bitmap)
                } else {
                    Log.e("CameraX", "Bitmap conversion failed")
                }
                imageProxy.close()
            }

            cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalysis
            )
        }, ContextCompat.getMainExecutor(context))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

/**
 * Convert ImageProxy to Bitmap
 */
fun ImageProxy.toBitmap(): Bitmap? {
    return try {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    } catch (e: Exception) {
        e.printStackTrace()
        null
    }
}
