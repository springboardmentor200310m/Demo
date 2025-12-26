import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes):
    """
    Build a CNN model for instrument classification.
    
    Args:
        input_shape (tuple): Shape of the input spectrogram (height, width, channels).
        num_classes (int): Number of instrument classes.
        
    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax') # Use 'sigmoid' for multi-label
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # Use 'binary_crossentropy' for multi-label
                  metrics=['accuracy'])
    
    return model
