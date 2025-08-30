package com.example.strawberrypickerapp

import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel

class Classifier(assetManager: AssetManager) {

    private var interpreter: Interpreter
    private val imageSize = 150

    init {
        // Load TFLite model from assets
        val fileDescriptor = assetManager.openFd("strawberry_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(buffer)
    }

    fun classify(bitmap: Bitmap): String {
        // Resize the image to match the model input
        val resized = Bitmap.createScaledBitmap(bitmap, imageSize, imageSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap pixels to float values
        val pixels = IntArray(imageSize * imageSize)
        resized.getPixels(pixels, 0, imageSize, 0, 0, imageSize, imageSize)
        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        // Prepare output array
        val output = Array(1) { FloatArray(1) }
        interpreter.run(inputBuffer, output)

        // Return result
        return if (output[0][0] > 0.5f) "✅ Good (Pick)" else "❌ Not Good (Unpick)"
    }
}
