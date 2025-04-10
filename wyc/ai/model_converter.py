import __main__
import tensorflow as tf # type: ignore
import os
import sys

def convert_h5_to_tflite(h5_model_path, tflite_model_path=None):
    """
    Convert a Keras H5 model to TFLite format.
    
    Args:
        h5_model_path: Path to the H5 model file
        tflite_model_path: Path where to save the TFLite model (optional)
                          If not provided, will use the same path but with .tflite extension
    
    Returns:
        Path to the saved TFLite model
    """
    # Check if the H5 model exists
    if not os.path.exists(h5_model_path):
        print(f"Error: H5 model not found at {h5_model_path}")
        return None
    
    # Create default TFLite model path if not provided
    if tflite_model_path is None:
        tflite_model_path = os.path.splitext(h5_model_path)[0] + '.tflite'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)
    
    try:
        print(f"Loading H5 model from {h5_model_path}...")
        model = tf.keras.models.load_model(h5_model_path)
        
        print("Converting model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model
        print(f"Saving TFLite model to {tflite_model_path}...")
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"Model successfully converted and saved:")
        print(f"  • Original H5 model: {h5_model_path}")
        print(f"  • TFLite model size: {os.path.getsize(tflite_model_path) / (1024 * 1024):.2f} MB")
        print(f"  • TFLite model path: {tflite_model_path}")
        
        return tflite_model_path
    
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Define model paths
    model_path = "models/2/food_ingredient_model_best.h5"
    tflite_model_path = "models/2/food_ingredient_model_best.tflite"
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if len(sys.argv) > 2:
            tflite_model_path = sys.argv[2]
    
    # Convert the model
    convert_h5_to_tflite(model_path, tflite_model_path)
