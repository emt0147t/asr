# Keep native JNI methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep ONNX Runtime classes
-keep class ai.onnxruntime.** { *; }

# Keep app classes
-keep class com.example.synthetic_kws.** { *; }
