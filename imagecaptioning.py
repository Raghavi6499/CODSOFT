import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import requests
from io import BytesIO
from collections import Counter
import warnings
import gc
warnings.filterwarnings('ignore')

# Configure TensorFlow for low memory usage
print("💻 Using CPU (tensorflow-cpu)")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

# Define and register the custom function
@tf.keras.utils.register_keras_serializable()
def repeat_context(inputs):
    context, caption = inputs
    seq_len = tf.shape(caption)[1]
    context = tf.expand_dims(context, axis=1)
    context = tf.tile(context, [1, seq_len, 1])
    return context

class ImprovedLightweightImageCaptioning:
    """Improved Lightweight Image Captioning Model - Fixed tokenization issues"""

    def __init__(self, vocab_size=300, embedding_dim=64, units=128, max_length=10):
        """Initialize with optimized parameters"""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.max_length = max_length
        self.tokenizer = None
        self.feature_extractor = None
        self.decoder = None
        self.model = None

        print(f"🐼 Improved Lightweight Image Captioning Model")
        print(f"   Vocab Size: {vocab_size}")
        print(f"   Embedding Dim: {embedding_dim}")
        print(f"   Hidden Units: {units}")
        print(f"   Max Length: {max_length}")

    def build_lightweight_feature_extractor(self):
        """Build feature extractor using VGG16"""
        print("🔧 Building feature extractor (VGG16)...")

        # Use VGG16 as pre-trained model
        base_model = tf.keras.applications.VGG16(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze the base model to save memory
        base_model.trainable = False

        # Add lightweight processing layers
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        self.feature_extractor = tf.keras.Model(inputs, x, name='lightweight_feature_extractor')
        print("✅ Feature extractor built successfully!")
        return self.feature_extractor

    def build_lightweight_decoder(self):
        """Build improved LSTM-based decoder"""
        print("🔧 Building improved lightweight decoder...")

        # Input layers
        image_features = tf.keras.Input(shape=(self.embedding_dim,), name='image_features')
        caption_input = tf.keras.Input(shape=(None,), dtype='int32', name='caption_input')

        # Embedding layer for captions
        embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            mask_zero=False  # Keep False to prevent BroadcastTo error
        )(caption_input)

        # LSTM layer
        lstm1 = tf.keras.layers.LSTM(
            self.units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )(embedding)

        # Process image features
        image_context = tf.keras.layers.Dense(self.units, activation='relu')(image_features)

        # Dynamic repeat to match current sequence length
        image_context = tf.keras.layers.Lambda(repeat_context, name='repeat_context')([image_context, caption_input])

        # Combine with image context
        combined = tf.keras.layers.Concatenate(axis=-1)([lstm1, image_context])

        # Second LSTM layer
        lstm2 = tf.keras.layers.LSTM(
            self.units // 2,
            return_sequences=True,
            dropout=0.2
        )(combined)

        # Output layer
        outputs = tf.keras.layers.Dense(self.vocab_size, activation='softmax')(lstm2)

        self.decoder = tf.keras.Model(
            [caption_input, image_features],
            outputs,
            name='lightweight_decoder'
        )
        print("✅ Improved lightweight decoder built successfully!")
        return self.decoder

    def build_complete_model(self):
        """Build the complete lightweight model"""
        tf.keras.backend.clear_session()
        print("🔧 Building complete improved model...")

        # Build components
        self.build_lightweight_feature_extractor()
        self.build_lightweight_decoder()

        # Complete model inputs
        image_input = tf.keras.Input(shape=(224, 224, 3), name='image')
        caption_input = tf.keras.Input(shape=(None,), dtype='int32', name='caption')

        # Forward pass
        image_features = self.feature_extractor(image_input)
        caption_output = self.decoder([caption_input, image_features])

        self.model = tf.keras.Model(
            [image_input, caption_input],
            caption_output,
            name='improved_lightweight_captioning'
        )

        print("✅ Complete improved model built successfully!")
        print(f"📊 Model parameters: {self.model.count_params():,}")
        return self.model

    def create_improved_dataset(self):
        """Create improved sample dataset with better variety"""
        print("📝 Creating improved sample dataset...")

        sample_captions = [
            "a red horizontal stripe appears",
            "green vertical stripe is visible",
            "blue square shape shows",
            "yellow square appears bright",
            "cyan horizontal band visible",
            "red stripe runs across image",
            "green stripe goes vertically",
            "blue colored square present",
            "yellow bright square shows",
            "cyan band appears horizontal",
            "colorful red stripe visible",
            "bright green vertical line",
            "dark blue square shape",
            "light yellow square area",
            "pale cyan horizontal stripe",
            "vivid red horizontal band",
            "deep green vertical stripe",
            "solid blue square block",
            "golden yellow square patch",
            "turquoise horizontal stripe"
        ]

        # Add start and end tokens
        processed_captions = [f"<start> {caption} <end>" for caption in sample_captions]
        return processed_captions

    def build_improved_tokenizer(self, captions):
        """Build improved tokenizer with better vocabulary management"""
        print("🔤 Building improved tokenizer...")

        # Create vocabulary
        all_words = []
        for caption in captions:
            words = caption.lower().split()
            all_words.extend(words)

        # Count frequencies and create vocab with essential tokens first
        word_counts = Counter(all_words)

        # Essential tokens
        essential_tokens = ['<pad>', '<unk>', '<start>', '<end>']

        # Content words (excluding essential tokens that are already added)
        content_words = []
        for word, count in word_counts.most_common():
            if word not in essential_tokens and len(content_words) < (self.vocab_size - len(essential_tokens)):
                content_words.append(word)

        vocab = essential_tokens + content_words

        # Create mappings
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for word, idx in word_to_idx.items()}

        self.tokenizer = {
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab_size': len(vocab)
        }

        print(f"✅ Improved tokenizer built with {len(vocab)} words")
        print(f"📝 Essential tokens: {essential_tokens}")
        print(f"📝 Sample content words: {content_words[:10]}")
        return self.tokenizer

    def encode_captions(self, captions):
        """Encode captions to sequences"""
        encoded_captions = []

        for caption in captions:
            words = caption.lower().split()
            encoded = []
            for word in words:
                if word in self.tokenizer['word_to_idx']:
                    encoded.append(self.tokenizer['word_to_idx'][word])
                else:
                    encoded.append(self.tokenizer['word_to_idx']['<unk>'])
            encoded_captions.append(encoded)

        # Pad sequences
        encoded_captions = tf.keras.preprocessing.sequence.pad_sequences(
            encoded_captions,
            maxlen=self.max_length,
            padding='post',
            value=self.tokenizer['word_to_idx']['<pad>']
        )

        return encoded_captions

    def create_improved_images(self, num_images=20):
        """Create improved sample images with clearer patterns"""
        print(f"🖼️ Creating {num_images} improved sample images...")

        images = []
        for i in range(num_images):
            # Create images with darker background for better contrast
            img = np.random.rand(224, 224, 3).astype(np.float32) * 0.2  # Darker background

            # Add very clear, distinctive patterns
            pattern_type = i % 5

            if pattern_type == 0:  # Red horizontal stripe
                img[80:140, :, :] = [0.9, 0.1, 0.1]
            elif pattern_type == 1:  # Green vertical stripe
                img[:, 80:140, :] = [0.1, 0.9, 0.1]
            elif pattern_type == 2:  # Blue square
                img[70:150, 70:150, :] = [0.1, 0.1, 0.9]
            elif pattern_type == 3:  # Yellow square
                img[60:130, 60:130, :] = [0.9, 0.9, 0.1]
            else:  # Cyan horizontal band
                img[50:100, :, :] = [0.1, 0.8, 0.8]

            images.append(img)

        return np.array(images, dtype=np.float32)

    def prepare_training_data(self, images, captions):
        """Prepare training data with memory optimization"""
        print("📊 Preparing training data...")

        # Validate input shapes
        if images.shape[1:] != (224, 224, 3):
            raise ValueError(f"Expected image shape (224, 224, 3), got {images.shape[1:]}")

        encoded_captions = self.encode_captions(captions)
        input_sequences = encoded_captions[:, :-1]
        target_sequences = encoded_captions[:, 1:]

        print(f"📐 Data shapes:")
        print(f"   Images: {images.shape}")
        print(f"   Input sequences: {input_sequences.shape}")
        print(f"   Target sequences: {target_sequences.shape}")

        return images, input_sequences, target_sequences

    def compile_model(self):
        """Compile model with optimized settings"""
        print("⚙️ Compiling model...")

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("✅ Model compiled successfully!")

    def train_model(self, images, input_sequences, target_sequences, epochs=50, batch_size=4):
        """Train model with improved settings"""
        print(f"🚀 Starting training (epochs={epochs}, batch_size={batch_size})...")

        # Check memory
        memory_info = get_memory_info()
        if memory_info and memory_info['available'] < 2.0:
            print(f"⚠️ Low memory warning: {memory_info['available']:.1f} GB available")

        # Improved callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        # Train with validation split
        history = self.model.fit(
            [images, input_sequences],
            target_sequences,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            validation_split=0.2
        )

        tf.keras.backend.clear_session()
        gc.collect()
        print("✅ Training completed!")
        return history

    def generate_caption(self, image, max_length=None, use_sampling=True):
        """Improved caption generation with sampling"""
        if max_length is None:
            max_length = self.max_length

        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Extract features
        image_features = self.feature_extractor(image)

        # Initialize caption
        caption = [self.tokenizer['word_to_idx']['<start>']]

        for step in range(max_length):
            # Prepare input
            caption_input = tf.expand_dims(caption, axis=0)

            # Predict next word
            predictions = self.decoder([caption_input, image_features])
            next_word_probs = predictions[0, -1, :]

            if use_sampling:
                # Use temperature sampling for more variety
                temperature = 0.7
                next_word_probs = next_word_probs / temperature
                next_word_probs = tf.nn.softmax(next_word_probs).numpy()

                # Sample from top-k predictions (avoid pad and unk if possible)
                top_k = 5
                top_indices = np.argsort(next_word_probs)[-top_k:]

                # Filter out pad and unk tokens if we have other options
                filtered_indices = []
                for idx in top_indices:
                    word = self.tokenizer['idx_to_word'].get(idx, '<unk>')
                    if word not in ['<pad>', '<unk>'] or len(filtered_indices) == 0:
                        filtered_indices.append(idx)

                if filtered_indices:
                    top_indices = filtered_indices

                top_probs = next_word_probs[top_indices]
                top_probs = top_probs / np.sum(top_probs)

                predicted_id = np.random.choice(top_indices, p=top_probs)
            else:
                # Use greedy decoding
                predicted_id = tf.argmax(next_word_probs).numpy()

            # Add to caption
            caption.append(predicted_id)

            # Stop if end token
            if predicted_id == self.tokenizer['word_to_idx']['<end>']:
                break

        # Convert to words
        caption_words = []
        for idx in caption[1:]:  # Skip start token
            if idx == self.tokenizer['word_to_idx']['<end>']:
                break
            word = self.tokenizer['idx_to_word'].get(idx, '<unk>')
            if word not in ['<pad>', '<start>']:
                caption_words.append(word)

        # Return meaningful caption or fallback
        if caption_words and not all(word == '<unk>' for word in caption_words):
            return ' '.join(caption_words)
        else:
            return "colorful pattern visible"

    def save_model(self, filepath):
        """Save model components"""
        print(f"💾 Saving model to {filepath}...")

        try:
            self.model.save(f"{filepath}_model.keras")
            with open(f"{filepath}_tokenizer.json", 'w') as f:
                json.dump(self.tokenizer, f, indent=2)
            print("✅ Model saved successfully!")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def load_model(self, filepath):
        """Load model components"""
        print(f"📂 Loading model from {filepath}...")

        try:
            model_path = f"{filepath}_model.keras"
            tokenizer_path = f"{filepath}_tokenizer.json"

            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Model or tokenizer file not found at {filepath}")

            self.model = tf.keras.models.load_model(model_path, custom_objects={'repeat_context': repeat_context})
            self.feature_extractor = self.model.get_layer('lightweight_feature_extractor')
            self.decoder = self.model.get_layer('lightweight_decoder')

            with open(tokenizer_path, 'r') as f:
                self.tokenizer = json.load(f)

            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def visualize_predictions(self, images, captions, num_samples=3):
        """Visualize predictions"""
        plt.figure(figsize=(15, 5 * num_samples))

        for i in range(min(num_samples, len(images))):
            generated_caption = self.generate_caption(images[i])

            plt.subplot(num_samples, 1, i + 1)
            plt.imshow(images[i])
            plt.title(f"Original: {captions[i]}\nGenerated: {generated_caption}", fontsize=12)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, images, captions):
        """Evaluate model with multiple generation attempts"""
        print("📊 Evaluating model...")

        for i, image in enumerate(images[:5]):
            print(f"Image {i+1}:")
            print(f"  Original: {captions[i] if i < len(captions) else 'N/A'}")

            # Generate multiple captions to show variety
            for attempt in range(3):
                generated = self.generate_caption(image, use_sampling=True)
                print(f"  Generated {attempt+1}: {generated}")
            print()

class ImageUtils:
    @staticmethod
    def load_image_from_path(path, target_size=(224, 224)):
        """Load and resize image from path"""
        try:
            image = Image.open(path)
            image = image.convert('RGB')
            image = image.resize(target_size)
            return np.array(image, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def load_image_from_url(url, target_size=(224, 224)):
        """Load and resize image from URL"""
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image = image.resize(target_size)
            return np.array(image, dtype=np.float32) / 255.0
        except Exception as e:
            print(f"Error loading image from URL: {e}")
            return None

def train_improved_model():
    """Train the improved model"""
    print("🚀 Training Improved Lightweight Image Captioning Model")
    print("Fixed tokenization and generation issues")
    print("="*60)

    # Initialize model with optimized parameters
    model = ImprovedLightweightImageCaptioning(
        vocab_size=300,
        embedding_dim=64,
        units=128,
        max_length=10
    )

    # Create improved data
    print("\n📝 Step 1: Creating improved dataset...")
    captions = model.create_improved_dataset()
    images = model.create_improved_images(len(captions))

    # Build tokenizer
    print("\n🔤 Step 2: Building improved tokenizer...")
    model.build_improved_tokenizer(captions)

    # Build model
    print("\n🔧 Step 3: Building improved model...")
    model.build_complete_model()

    # Show model summary
    print("\n📊 Model Summary:")
    model.model.summary()

    # Prepare data
    print("\n📊 Step 4: Preparing training data...")
    train_images, input_sequences, target_sequences = model.prepare_training_data(images, captions)

    # Compile model
    print("\n⚙️ Step 5: Compiling model...")
    model.compile_model()

    # Train model
    print("\n🚀 Step 6: Training model...")
    history = model.train_model(
        train_images,
        input_sequences,
        target_sequences,
        epochs=30,
        batch_size=4
    )

    # Plot results
    print("\n📈 Step 7: Plotting training history...")
    model.plot_training_history(history)

    # Evaluate
    print("\n📊 Step 8: Evaluating model...")
    model.evaluate_model(images[:3], captions[:3])

    # Visualize
    print("\n🖼️ Step 9: Visualizing predictions...")
    model.visualize_predictions(images, captions, num_samples=3)

    # Save model
    print("\n💾 Step 10: Saving model...")
    model.save_model("improved_lightweight_captioning")

    print("\n✅ Training completed successfully!")
    return model

def test_improved_model():
    """Test the improved model"""
    print("🧪 Testing Improved Model")
    print("="*30)

    model = ImprovedLightweightImageCaptioning()

    try:
        model.load_model("improved_lightweight_captioning")
        print("✅ Model loaded successfully!")

        # Test with new images
        test_images = model.create_improved_images(5)

        print("\n🖼️ Generating captions for test images...")
        for i, image in enumerate(test_images):
            print(f"\nImage {i+1}:")

            # Generate multiple captions to show variety
            for attempt in range(3):
                caption = model.generate_caption(image, use_sampling=True)
                print(f"  Attempt {attempt+1}: {caption}")

            plt.figure(figsize=(6, 4))
            plt.imshow(image)
            plt.title(f"Test Image {i+1}")
            plt.axis('off')
            plt.show()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Training new improved model...")
        return train_improved_model()

def get_memory_info():
    """Get system memory information"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024**3),
            'available': memory.available / (1024**3),
            'used': memory.used / (1024**3),
            'percent': memory.percent
        }
    except ImportError:
        return None

def main():
    """Main function"""
    print("🐼 Improved Lightweight Image Captioning AI")
    print("Fixed tokenization and generation issues")
    print("="*50)

    print(f"TensorFlow version: {tf.__version__}")

    # Check available memory
    memory_info = get_memory_info()
    if memory_info:
        print(f"Available RAM: {memory_info['available']:.1f} GB")

    while True:
        print("\n" + "="*40)
        print("Improved Model Menu:")
        print("1. Train new improved model")
        print("2. Test saved improved model")
        print("3. Quick demo")
        print("4. Exit")
        print("="*40)

        choice = input("Enter choice (1-4): ").strip()

        if choice == '1':
            train_improved_model()

        elif choice == '2':
            test_improved_model()

        elif choice == '3':
            # Quick demo
            model = ImprovedLightweightImageCaptioning()

            try:
                model.load_model("improved_lightweight_captioning")
            except:
                print("Training new model for demo...")
                model = train_improved_model()

            # Demo with multiple attempts
            images = model.create_improved_images(3)

            plt.figure(figsize=(12, 4))
            for i, image in enumerate(images):
                caption = model.generate_caption(image, use_sampling=True)

                plt.subplot(1, 3, i + 1)
                plt.imshow(image)
                plt.title(f"Generated:\n{caption}")
                plt.axis('off')

                print(f"Image {i+1}: {caption}")

            plt.tight_layout()
            plt.show()

        elif choice == '4':
            print("👋 Goodbye!")
            break

        else:
            print("❌ Invalid choice!")

        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
