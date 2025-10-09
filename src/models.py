import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np # Wird für tf.Variable Initialisierung benötigt

from .config import LEARNING_RATE
# HypersphereNormalization bleibt unverändert, da sie für den Encoder-Output vor der Quantisierung nützlich sein könnte
class HypersphereNormalization(layers.Layer):
    def __init__(self, **kwargs):
        super(HypersphereNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        norm = tf.norm(inputs, ord='euclidean', axis=-1, keepdims=True)
        # Add epsilon to prevent division by zero
        normalized_output = inputs / (norm + tf.keras.backend.epsilon())
        return normalized_output

# ---- Vector Quantizer Layer ----
class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost  # Beta-Parameter

        # Initialize embeddings with a glorot_uniform distribution
        # You can choose other initializers as well
        # Using tf.Variable for embeddings allows them to be directly updated
        # by the gradient of the decoder's loss (straight-through estimator).
        initializer = tf.keras.initializers.GlorotUniform()
        self.embeddings = tf.Variable(
            initializer(shape=(self.num_embeddings, self.embedding_dim)),
            trainable=True,
            name="embeddings_codebook"
        )

    def call(self, inputs):
        # inputs shape: (batch_size, sequence_length_after_convs, latent_dim)
        # We need to reshape inputs to (batch_size * sequence_length_after_convs, latent_dim)
        # to find the closest embedding for each position in the sequence.
        
        # Flatten the input to (num_vectors, embedding_dim)
        flat_inputs = tf.reshape(inputs, [-1, self.embedding_dim])

        # Calculate distances (squared Euclidean distance)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x * y^T
        distances = (
            tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=1)
            - 2 * tf.matmul(flat_inputs, self.embeddings, transpose_b=True)
        )

        # Get the indices of the closest embedding vectors
        encoding_indices = tf.argmin(distances, axis=1)

        # Convert indices to one-hot encodings
        one_hot_encodings = tf.one_hot(encoding_indices, self.num_embeddings, dtype=distances.dtype)

        # Quantize the input: lookup the embeddings using the one-hot encodings
        quantized_latents = tf.matmul(one_hot_encodings, self.embeddings)
        
        # Reshape back to original input shape
        quantized_latents = tf.reshape(quantized_latents, tf.shape(inputs))

        # Calculate VQ-VAE losses
        # Commitment loss (Encoders output pulled towards the chosen embedding)
        # L_commitment = ||sg[z_e(x)] - e_i||^2
        # `inputs` (z_e(x)) are detached from the gradient flow (stop_gradient)
        # `quantized_latents` (e_i) are differentiable and updated by the decoder's loss
        commitment_loss = self.commitment_cost * tf.reduce_mean(
            (tf.stop_gradient(quantized_latents) - inputs)**2
        )
        
        # Embedding loss (Codebook embeddings updated based on encoder's output)
        # L_embedding = ||z_e(x) - sg[e_i]||^2
        # `inputs` (z_e(x)) are differentiable
        # `quantized_latents` (e_i) are detached from the gradient flow
        embedding_loss = tf.reduce_mean(
            (inputs - tf.stop_gradient(quantized_latents))**2
        )

        # Add losses to the model. This is crucial for custom training loops or when
        # using model.compile/fit without explicitly defining a custom training step.
        self.add_loss(commitment_loss)
        self.add_loss(embedding_loss)

        # Straight-Through Estimator:
        # In the backward pass, gradients are copied from the output of the VQ layer (quantized_latents)
        # to the input of the VQ layer (inputs). This effectively allows gradients to flow
        # through the non-differentiable quantization step.
        quantized_latents = inputs + tf.stop_gradient(quantized_latents - inputs)
        
        return quantized_latents

    def get_code_indices(self, flat_inputs):
        # This method is useful for inference/visualization to get the chosen codebook indices
        distances = (
            tf.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=1)
            - 2 * tf.matmul(flat_inputs, self.embeddings, transpose_b=True)
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "commitment_cost": self.commitment_cost,
        })
        return config

# ---- VQ-VAE Model Definition ----
class VQVAE(Model):
    def __init__(self, input_shape, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Encoder Architecture (adapted from your ConvAutoEncoder)
        # The last Conv1D layer outputs features with `latent_dim` channels,
        # which will be the `embedding_dim` for the VQ layer.
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=8, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2, padding="same"),
            layers.Conv1D(filters=32, kernel_size=4, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2, padding="same"),
            # The output of this layer will be fed to the VQ layer.
            # Its last dimension should match `embedding_dim`.
            layers.Conv1D(filters=latent_dim, kernel_size=3, activation="relu", padding="same", name="encoder_output_pre_vq"),
            # HypersphereNormalization could still be here, but usually,
            # VQ-VAE's don't normalize the input to the VQ layer to a unit sphere.
            # It's more common for the VQ layer to learn embeddings anywhere.
            # I'll remove it for typical VQ-VAE behavior.
            # If you want to keep it, you should experiment carefully.
            # HypersphereNormalization() # Removed for standard VQ-VAE implementation
        ], name='encoder')

        # VQ Layer
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, name="vector_quantizer")

        # Decoder Architecture (adapted from your ConvAutoEncoder)
        # The decoder takes the quantized latents as input.
        # Ensure the input shape to the decoder matches the output shape of `quantized_latents` from VQ layer.
        
        # To calculate the required input shape for the decoder's first Conv1DTranspose layer:
        # You need to run a dummy input through the encoder and VQ layer to get the shape.
        # This is a bit tricky for Keras Sequential model definition.
        # A common approach is to use a functional API for the decoder if precise shape handling is needed,
        # or calculate the shape based on the encoder's downsampling.
        
        # Let's estimate the shape for `Conv1DTranspose` based on `MaxPooling1D`
        # Input length: `input_shape[0]`
        # After 1st MaxPooling1D(pool_size=2): `input_shape[0] / 2`
        # After 2nd MaxPooling1D(pool_size=2): `input_shape[0] / 4`
        # So the length of the sequence fed to the VQ layer will be `input_shape[0] // 4`.
        # The VQ layer outputs the same spatial dimensions as its input.
        
        # So, the quantized_latents will have shape `(batch_size, input_shape[0] // 4, latent_dim)`
        
        # First Conv1DTranspose should have strides=2, padding="same" to double the size.
        # (input_length / 4) * 2 = input_length / 2
        # Second Conv1DTranspose with strides=2, padding="same" to double again.
        # (input_length / 2) * 2 = input_length
        
        self.decoder = tf.keras.Sequential([
            # Input to decoder is `(sequence_length_after_convs, latent_dim)`
            # We don't need an explicit Input layer here if this is part of a larger Model
            # but for a Sequential model, it's good practice.
            # However, when using it within `call`, the input will directly be `quantized_latents`.
            
            # The first Conv1DTranspose needs to know the filters and kernel_size,
            # but its input shape is dynamic, coming from `vq_layer`.
            # Keras handles this dynamically often, but if issues arise,
            # you might need to use the Functional API or pass `input_shape` here carefully.
            
            # Example: Decoder's first layer will receive (batch, original_length // 4, latent_dim)
            # Its output should be (batch, original_length // 2, some_filters)
            layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same"),
            # Its output should be (batch, original_length, some_filters)
            layers.Conv1DTranspose(filters=64, kernel_size=4, strides=2, activation="relu", padding="same"),
            # Final Conv1D to match original input channels
            layers.Conv1D(filters=input_shape[-1], kernel_size=8, activation="sigmoid", padding="same")
            # Using 'sigmoid' for reconstruction is good if your EKG data is normalized to [0, 1].
            # If it's normalized to [-1, 1] or has arbitrary range, 'tanh' or no activation might be better.
            # Since you're using `snippet / max_abs_val`, sigmoid is probably appropriate if `max_abs_val`
            # ensures values are within a range that `sigmoid` can cover effectively.
        ], name='decoder')

    def call(self, x):
        encoder_output = self.encoder(x)
        quantized_latents = self.vq_layer(encoder_output)
        reconstructions = self.decoder(quantized_latents)
        
        # Add the reconstruction loss. This is the third loss component.
        # The VQ-VAE paper typically uses MSE.
        reconstruction_loss = tf.reduce_mean((x - reconstructions)**2, name="reconstruction_loss")
        self.add_loss(reconstruction_loss)
        
        return reconstructions

    def get_latent_representation(self, x):
        # This method can be used to get the actual quantized embeddings for visualization
        encoder_output = self.encoder(x)
        # Use the VQ layer's call method for inference to get the quantized latents
        # The call method returns (quantized_latents, loss) during training,
        # but for simple inference, you just want the first element.
        # However, the VQ layer also adds loss to the model. You might want
        # a specific inference path for the VQ layer if you want to avoid adding loss
        # during this call (e.g., if you're only evaluating a trained model).
        
        # For simplicity, let's assume `vq_layer.call` also returns just the quantized_latents
        # when called outside the main training loop context (no gradients).
        
        # For a clean separation, you might make a dedicated inference model:
        # self.inference_encoder = Model(inputs=self.encoder.input,
        #                                outputs=self.vq_layer(self.encoder.output))
        # Then call self.inference_encoder.predict(x)
        
        # A simpler way for this method: Directly get the quantized latents from the VQ layer's logic
        flat_encoded_features = tf.reshape(encoder_output, [-1, self.latent_dim])
        code_indices = self.vq_layer.get_code_indices(flat_encoded_features)
        
        # Get the actual embeddings based on indices
        quantized_output_flat = tf.gather(self.vq_layer.embeddings, code_indices)
        
        # Reshape back to the original spatial dimensions of the encoder output
        quantized_output_reshaped = tf.reshape(quantized_output_flat, tf.shape(encoder_output))
        
        return quantized_output_reshaped

    def get_codebook_embeddings(self):
        # Returns the current state of the codebook embeddings
        return self.vq_layer.embeddings


# Die Funktion zum Trainieren und Evaluieren des VQ-VAE
def train_and_evaluate_vqvae(train_data, test_data, input_shape, latent_dim, num_embeddings, commitment_cost, epochs=10, batch_size=32):
    # Instanziieren des VQ-VAE Modells
    vq_vae = VQVAE(input_shape, latent_dim, num_embeddings, commitment_cost)
    
    # Der Optimizer muss hier auf das VQ-VAE angewendet werden.
    # Die Loss-Berechnung ist in der `call`-Methode der VQVAE-Klasse (`add_loss`)
    # sowie in der `VectorQuantizer`-Schicht enthalten.
    
    # Da die Losses mit `add_loss` hinzugefügt werden, kompiliert Keras sie automatisch.
    # Sie können hier den `loss='mse'` für den Rekonstruktionsverlust angeben,
    # obwohl der Hauptrekonstruktionsverlust manuell in `call` hinzugefügt wird.
    # Wenn Sie nur die `add_loss` Methode verwenden, können Sie `loss=None` setzen
    # und Keras wird nur die hinzugefügten Verluste optimieren.
    
    # Für Klarheit: Stellen Sie sicher, dass der `compile` Schritt das Modell für das Training vorbereitet.
    # Da `add_loss` verwendet wird, wird das Modell korrekt trainiert.



    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    vq_vae.compile(optimizer=optimizer, loss = 'mse') # Keras will automatically pick up losses added via add_loss() , mse=mean sqaured error
    
    print(f"Starte Training für VQ-VAE mit Latenz-Dimension: {latent_dim}, Codebook-Größe: {num_embeddings}")
    
    # Der `fit` Aufruf bleibt ähnlich, da Keras die Losses intern verwaltet.
    history = vq_vae.fit(train_data, train_data, 
                          epochs=epochs, 
                          batch_size=batch_size, 
                          validation_data=(test_data, test_data), 
                          verbose=1) # Set verbose to 1 to see progress

    # Evaluierung: Keras `evaluate` wird alle Losses summieren.
    # Der zurückgegebene 'loss' Wert ist die Summe aus Rekonstruktions-Loss, Commitment-Loss und Embedding-Loss.
    total_loss = vq_vae.evaluate(test_data, test_data, verbose=0)
    print(f"Validierungs-Gesamtfehler (Summe der Losses) für VQ-VAE: {total_loss:.4f}")
    
    # Um nur den Rekonstruktionsfehler zu sehen, müssten Sie eine benutzerdefinierte Metrik
    # oder eine separate Vorwärts-Pass-Funktion definieren, die nur den Rekonstruktionsfehler berechnet.
    # Für VQ-VAE-Evaluierung ist es oft hilfreich, alle Komponenten des Verlusts zu verfolgen.
    
    # Optional: Den Rekonstruktionsfehler separat berechnen
    # reconstructions = vq_vae.predict(test_data)
    # reconstruction_mse = tf.reduce_mean((test_data - reconstructions)**2).numpy()
    # print(f"Validierungs-Rekonstruktionsfehler (MSE) für VQ-VAE: {reconstruction_mse:.4f}")

    return vq_vae, history, total_loss # Oder history, reconstruction_mse
