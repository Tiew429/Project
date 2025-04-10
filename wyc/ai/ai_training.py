import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import subprocess
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import json
import zipfile
import shutil
from pathlib import Path

# Function to install required packages
def install_required_packages():
    print("Checking required packages...")
    packages = {
        "tensorflow": "tensorflow",
        "opencv-python": "cv2",
        "patool": "patoolib",
        "datasets": "datasets",
        "kagglehub": "kagglehub",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib"
    }
    
    # Check and install missing packages
    missing_packages = []
    for package_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is already installed")
        except ImportError:
            missing_packages.append(package_name)
    
    # Install missing packages if any
    if missing_packages:
        print(f"Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
        print("All required packages installed successfully!")
    else:
        print("All required packages are already installed!")
        
    # Import necessary modules after ensuring installation
    global tf, cv2
    import tensorflow as tf # type: ignore
    import cv2 # type: ignore

# Function to download and extract the dataset
def download_food_ingredients_dataset():
    print("Checking for food ingredients dataset...")
    import kagglehub # type: ignore
    
    try:
        # Download latest version using kagglehub (similar to the notebook)
        print("Downloading food ingredients dataset...")
        dataset_path = kagglehub.dataset_download("daffaff/ingredients-food-dataset")
        
        print(f"Dataset downloaded successfully to: {dataset_path}")
        
        # The path to the actual dataset folder
        final_path = Path(dataset_path) / "Dataset_ingredients"
        
        # Verify dataset exists
        if not final_path.exists():
            print(f"Error: Dataset directory '{final_path}' not found after download")
            sys.exit(1)
            
        return final_path
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Please check your internet connection and try again.")
        sys.exit(1)

# Function to prepare the data for training
def prepare_data(dataset_path):
    import tensorflow as tf # type: ignore
    from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
    
    print("Preparing data for training...")
    
    # Parameters
    img_height, img_width = 224, 224
    batch_size = 32
    validation_split = 0.2
    test_split = 0.1
    
    # Create data generators with data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split+test_split
    )
    
    # Create train generator
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Temporary validation generator (includes test data)
    temp_val_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Split validation data into validation and test sets
    val_samples = int(len(temp_val_generator.filenames) * validation_split / (validation_split + test_split))
    test_samples = len(temp_val_generator.filenames) - val_samples
    
    # Create separate validation and test generators
    validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=test_split/(validation_split+test_split))
    
    validation_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=False
    )
    
    test_generator = validation_datagen.flow_from_directory(
        dataset_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    print(f"Training samples: {len(train_generator.filenames)}")
    print(f"Validation samples: {val_samples}")
    print(f"Testing samples: {test_samples}")
    
    return train_generator, validation_generator, test_generator, num_classes

# Function to build the model
def build_model(num_classes):
    import tensorflow as tf # type: ignore
    from tensorflow.keras.applications import MobileNetV2 # type: ignore
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
    from tensorflow.keras.models import Model # type: ignore
    
    print("Building the model...")
    
    # Base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model built successfully!")
    return model

# Function to load and plot training history from saved file
def load_and_plot_history():
    print("\nLoading training history...")
    history_path = 'model_history.json'
    
    if not os.path.exists(history_path):
        print("No training history file found. Plot cannot be displayed.")
        return False
    
    try:
        with open(history_path, 'r') as f:
            history_data = json.load(f)
        
        # Convert to history-like object
        class HistoryObject:
            pass
        
        history = HistoryObject()
        history.history = history_data
        
        # Plot the history
        plot_training_history(history)
        return True
    except Exception as e:
        print(f"Error loading training history: {str(e)}")
        return False

# Function to train the model
def train_model(model, train_generator, validation_generator):
    print("Training the model...")
    
    # Parameters
    epochs = 15  # Increased epochs to better show progress
    
    # Callbacks
    import tensorflow as tf # type: ignore
    from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping # type: ignore
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        'food_ingredient_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    # TensorBoard callback for visualization
    tensorboard_callback = TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Train the model with progress bar display
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint_callback, early_stopping, tensorboard_callback],
        verbose=1  # Show progress bar
    )
    
    print("Model training completed!")
    
    # Save the final model
    model.save('food_ingredient_model_final.keras')
    print("Model saved as 'food_ingredient_model_final.keras'")
    
    # Save history for later visualization
    with open('model_history.json', 'w') as f:
        json.dump(history.history, f)
    
    return history

# Function to evaluate the model
def evaluate_model(model, test_generator):
    print("\nEvaluating the model on test data...")
    
    # Evaluate the model on the test set with progress bar
    evaluation = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    
    print(f"\nTest loss: {evaluation[0]:.4f}")
    print(f"Test accuracy: {evaluation[1]:.4f}")
    
    # Make predictions on test data for a sample to show actual predictions
    print("\nGenerating predictions on some test samples...")
    
    # Get a batch of test data
    test_batch = next(test_generator)
    test_images, test_labels = test_batch
    
    # Get predictions
    predictions = model.predict(test_images[:5], verbose=1)  # Predict on first 5 images
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Display results for a few test images
    print("\nSample predictions:")
    for i in range(min(5, len(test_images))):
        true_class_idx = np.argmax(test_labels[i])
        predicted_class_idx = np.argmax(predictions[i])
        confidence = predictions[i][predicted_class_idx] * 100
        
        print(f"Sample {i+1}:")
        print(f"  True label: {class_names[true_class_idx]}")
        print(f"  Predicted: {class_names[predicted_class_idx]} (Confidence: {confidence:.2f}%)")
        print(f"  {'✓ Correct' if true_class_idx == predicted_class_idx else '✗ Incorrect'}")
    
    return evaluation

# Function to plot training history
def plot_training_history(history):
    print("Plotting training history...")
    
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("Training history plot saved as 'training_history.png'")

# Function to predict ingredients in an image
def predict_ingredients(model_path, image_path, class_names, top_n=5):
    import tensorflow as tf # type: ignore
    from tensorflow.keras.models import load_model # type: ignore
    from tensorflow.keras.preprocessing import image # type: ignore
    import numpy as np # type: ignore
    import cv2 # type: ignore
    
    print(f"\nAnalyzing image: {image_path}")
    
    try:
        # Load the trained model
        model = load_model(model_path)
        
        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            return
        
        # Resize the image
        img = cv2.resize(img, (224, 224))
        
        # Convert to RGB if needed (cv2 loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.predict(img, verbose=0)
        
        # Check if the number of predictions matches the class names
        if len(predictions[0]) != len(class_names):
            print(f"Warning: Model output size ({len(predictions[0])}) doesn't match class names ({len(class_names)})")
            print("This might indicate a model/class names mismatch.")
            
            # If model has more outputs than class names, use only the first len(class_names) outputs
            if len(predictions[0]) > len(class_names):
                print("Truncating predictions to match available class names.")
                # Create a mask of valid indices
                valid_indices = np.array([i for i in range(len(predictions[0])) if i < len(class_names)])
                # Get top N among valid indices
                top_indices = valid_indices[np.argsort(predictions[0][valid_indices])[-top_n:][::-1]]
            else:
                # If we have more class names than outputs, just use the model outputs
                print("Using available predictions only.")
                # Get top indices from available predictions
                top_indices = predictions[0].argsort()[-min(top_n, len(predictions[0])):][::-1]
        else:
            # Normal case - get top N predictions
            top_indices = predictions[0].argsort()[-top_n:][::-1]
        
        # Make sure none of the indices are out of range
        top_indices = [i for i in top_indices if i < len(class_names)]
        
        # Create predictions
        top_predictions = [(class_names[i], float(predictions[0][i] * 100)) for i in top_indices]
        
        # Create figure with subplots - one for image and one for predictions
        plt.figure(figsize=(14, 7))
        
        # Show the image on the left subplot
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Input Image")
        
        # Show bar chart of predictions on the right subplot
        plt.subplot(1, 2, 2)
        
        # Extract ingredients and confidences
        ingredients = [item[0] for item in top_predictions]
        confidences = [item[1] for item in top_predictions]
        
        # Colors for bars - more confidence = more saturated color
        colors = plt.cm.viridis(np.array(confidences)/100)
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(top_predictions)), confidences, color=colors)
        
        # Add percentage labels to the bars
        for i, (bar, confidence) in enumerate(zip(bars, confidences)):
            plt.text(min(confidence + 2, 95), bar.get_y() + bar.get_height()/2, 
                    f"{confidence:.1f}%", va='center', fontweight='bold')
        
        # Set labels and title
        plt.yticks(range(len(top_predictions)), ingredients)
        plt.xlabel("Confidence (%)")
        plt.title("Predicted Ingredients")
        plt.xlim(0, 105)  # Set x-axis limit to allow space for percentage labels
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
        
        # Also print the predictions in the console
        print("\nPredicted Ingredients:")
        for ingredient, confidence in top_predictions:
            # Create a visual percentage bar with filled and empty blocks
            bar_length = 50  # Total length of bar
            filled_length = int(round(bar_length * confidence / 100))
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Print with formatted percentage
            print(f"{ingredient:<20} : {confidence:>5.1f}% |{bar}|")
        
        return top_predictions
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# Main function
if __name__ == "__main__":
    print("="*50)
    print("Food Ingredient Recognition System")
    print("="*50)
    
    # Install required packages if needed
    install_required_packages()
    
    while True:
        # Simple UI for choosing mode
        print("\nChoose an option:")
        print("1. Train a new model")
        print("2. Predict ingredients in an image")
        print("3. Display training history")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-3): ")
        
        if choice == "1":
            print("\nStarting model training...")
            
            # Download dataset
            dataset_path = download_food_ingredients_dataset()
            
            # Prepare data
            train_generator, validation_generator, test_generator, num_classes = prepare_data(dataset_path)
            
            # Build model
            model = build_model(num_classes)
            
            # Train model
            history = train_model(model, train_generator, validation_generator)
            
            # Evaluate model
            evaluation = evaluate_model(model, test_generator)
            
            # Plot training history
            plot_training_history(history)
            
            print("\nFood Ingredient Recognition Model Training Completed!")
            print(f"Final test accuracy: {evaluation[1]:.4f}")
            print(f"Final test loss: {evaluation[0]:.4f}")
            
            # Save the class names for later prediction
            class_names = list(train_generator.class_indices.keys())
            with open('model_class_names.json', 'w') as f:
                json.dump(class_names, f)
                
            # Save results summary
            with open('model_results.json', 'w') as f:
                json.dump({
                    'test_accuracy': float(evaluation[1]),
                    'test_loss': float(evaluation[0]),
                    'val_accuracy': float(history.history['val_accuracy'][-1]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'num_classes': num_classes,
                    'class_names': class_names
                }, f)
            
            print("\nResults and model saved. You can now use the model for prediction.")
            
        elif choice == "2":
            print("\nPredicting ingredients in an image...")
            
            # Check if model exists
            model_path = 'food_ingredient_model_final.keras'
            if not os.path.exists(model_path):
                model_path = 'food_ingredient_model_best.h5'
                if not os.path.exists(model_path):
                    print("Error: No trained model found. Please train a model first (option 1).")
                    sys.exit(1)
            
            # Load the model to get the number of classes
            try:
                import tensorflow as tf # type: ignore
                from tensorflow.keras.models import load_model # type: ignore
                
                print(f"Loading model from {model_path}...")
                model = load_model(model_path)
                
                # Get the number of output classes from the model
                num_classes = model.output_shape[1]
                print(f"Model has {num_classes} output classes")
                
                # Load class names
                class_names_path = 'model_class_names.json'
                if os.path.exists(class_names_path):
                    with open(class_names_path, 'r') as f:
                        class_names = json.load(f)
                    print(f"Loaded {len(class_names)} class names from {class_names_path}")
                else:
                    # If class names file doesn't exist, try to extract from model_results.json
                    results_path = 'model_results.json'
                    if os.path.exists(results_path):
                        with open(results_path, 'r') as f:
                            results = json.load(f)
                            class_names = results.get('class_names', [])
                        print(f"Loaded {len(class_names)} class names from {results_path}")
                    else:
                        # If no class names are found, use dataset structure to get class names
                        print("No class names file found. Attempting to retrieve class names from dataset...")
                        try:
                            dataset_path = download_food_ingredients_dataset()
                            class_names = sorted(os.listdir(dataset_path))
                            # Save class names for future use
                            with open('model_class_names.json', 'w') as f:
                                json.dump(class_names, f)
                            print(f"Retrieved {len(class_names)} class names from dataset")
                        except Exception as e:
                            print(f"Error: Could not determine class names. {str(e)}")
                            sys.exit(1)
                
                # Check if class names match model output size
                if len(class_names) != num_classes:
                    print(f"Warning: Number of class names ({len(class_names)}) doesn't match model output size ({num_classes})")
                    if len(class_names) < num_classes:
                        # Try to get more class names from dataset
                        try:
                            dataset_path = download_food_ingredients_dataset()
                            dataset_classes = sorted(os.listdir(dataset_path))
                            if len(dataset_classes) >= num_classes:
                                class_names = dataset_classes[:num_classes]
                                print(f"Updated class names from dataset to match model ({len(class_names)} classes)")
                            else:
                                # Pad with generic names if needed
                                missing = num_classes - len(class_names)
                                class_names.extend([f"Unknown-{i+1}" for i in range(missing)])
                                print(f"Added {missing} generic class names to match model output size")
                        except Exception as e:
                            # Pad with generic names
                            missing = num_classes - len(class_names)
                            class_names.extend([f"Unknown-{i+1}" for i in range(missing)])
                            print(f"Added {missing} generic class names to match model output size")
                    else:
                        # Truncate class names to match model output
                        class_names = class_names[:num_classes]
                        print(f"Truncated class names to match model output size ({len(class_names)} classes)")
                    
                    # Save updated class names
                    with open('model_class_names.json', 'w') as f:
                        json.dump(class_names, f)
            
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            
            # Get image path from user
            image_path = input("Enter the path to your image (e.g., 'images/tomato.png'): ")
            
            # Ensure image exists
            if not os.path.exists(image_path):
                print(f"Error: Image not found at '{image_path}'")
                sys.exit(1)
            
            # Predict ingredients
            predict_ingredients(model_path, image_path, class_names)
        
        elif choice == "3":
            # Display training history
            if not load_and_plot_history():
                print("No training history available. Please train a model first (option 1).")
        
        elif choice == "0":
            print("Exiting the program...")
            break
        else:
            print("Invalid choice. Please enter 0, 1, 2, or 3.")
