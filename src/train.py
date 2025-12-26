import tensorflow as tf
import numpy as np
import os
from model import build_model
from preprocessing import load_audio, generate_mel_spectrogram

def create_dummy_dataset(num_samples=100, input_shape=(128, 128, 1), num_classes=5):
    """
    Create a dummy dataset for testing the training loop.
    """
    X = np.random.rand(num_samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    y = tf.keras.utils.to_categorical(y, num_classes)
    return X, y

def train(epochs=5, batch_size=32):
    """
    Train the model.
    """
    input_shape = (128, 128, 1) # Example shape
    num_classes = 5 # Example classes: Piano, Guitar, Drums, Violin, Flute
    
    # Load data (using dummy data for now)
    print("Loading data...")
    X_train, y_train = create_dummy_dataset(num_samples=200, input_shape=input_shape, num_classes=num_classes)
    X_val, y_val = create_dummy_dataset(num_samples=50, input_shape=input_shape, num_classes=num_classes)
    
    # Build model
    print("Building model...")
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Train
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )
    
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save('models/instrunet_model.h5')
    print("Model saved to models/instrunet_model.h5")
    
    return history

if __name__ == '__main__':
    train()
