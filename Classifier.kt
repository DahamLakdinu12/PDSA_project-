package com.example.strawberrypickerapp

import android.content.res.AssetManager
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.io.FileInputStream
import java.nio.channels.FileChannel
import java.util.LinkedList
import kotlin.math.abs

class Classifier(assetManager: AssetManager) {

    private var interpreter: Interpreter
    private var imageSizeX: Int
    private var imageSizeY: Int
    private val numChannels: Int = 3  // RGB
    private val queue: LinkedList<Bitmap> = LinkedList() // FIFO Queue
    private val maxQueueSize = 5

    init {
        // Load TFLite model from assets
        val fileDescriptor = assetManager.openFd("strawberry_model.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        // Create interpreter
        interpreter = Interpreter(buffer)

        // Read input shape from model (e.g., [1,150,150,3])
        val inputShape = interpreter.getInputTensor(0).shape()
        imageSizeX = inputShape[1]
        imageSizeY = inputShape[2]
    }

    // Add image to FIFO queue
    fun enqueueImage(bitmap: Bitmap) {
        if (queue.size >= maxQueueSize) {
            queue.poll() // remove oldest image
        }
        queue.add(bitmap)
    }

    // Process next image in FIFO
    fun classifyNext(): String {
        val bitmap = queue.poll() ?: return "⏳ Waiting for image..."
        return classify(bitmap)
    }

    fun classify(bitmap: Bitmap): String {
        // Resize the image to match the model input
        val resized = Bitmap.createScaledBitmap(bitmap, imageSizeX, imageSizeY, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * imageSizeX * imageSizeY * numChannels)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap pixels to float values
        val pixels = IntArray(imageSizeX * imageSizeY)
        resized.getPixels(pixels, 0, imageSizeX, 0, 0, imageSizeX, imageSizeY)

        var avgRed = 0f
        var avgGreen = 0f
        var avgBlue = 0f

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)

            avgRed += r
            avgGreen += g
            avgBlue += b
        }

        avgRed /= pixels.size
        avgGreen /= pixels.size
        avgBlue /= pixels.size

        // Simple "not strawberry" filter:
        // strawberries should be reddish, so if avg red is not dominant -> Not a strawberry
        if (avgRed < 0.3f || avgRed < avgGreen || avgRed < avgBlue) {
            return "🍓 Not a Strawberry"
        }

        // Prepare output array for binary model
        val output = Array(1) { FloatArray(1) }
        interpreter.run(inputBuffer, output)

        // Binary classification result
        return if (output[0][0] > 0.5f) {
            "✅ Pickable Strawberry"
        } else {
            "❌ Unpickable Strawberry"
        }
    }

    fun close() {
        interpreter.close()
    }
}
