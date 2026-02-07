import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np # Wird für tf.Variable Initialisierung benötigt

from .config import LEARNING_RATE
# Hypersphere wird aktuell nicht genutzt
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


        initializer = tf.keras.initializers.GlorotUniform()
        self.embeddings = self.add_weight(
            name="embeddings_codebook",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer=initializer,
            trainable=True,
        )

    def call(self, inputs):
       
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
            (tf.stop_gradient(inputs) - quantized_latents)**2
        )

        # Add losses to the model.
        self.add_loss(commitment_loss)
        self.add_loss(embedding_loss)

        # Straight-Through Estimator:
        quantized_latents = inputs + tf.stop_gradient(quantized_latents - inputs)
        
        return quantized_latents

    def get_code_indices(self, flat_inputs):
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
        
       
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(filters=64, kernel_size=8, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2, padding="same"),
            layers.Conv1D(filters=32, kernel_size=4, activation="relu", padding="same"),
            layers.MaxPooling1D(pool_size=2, padding="same"),
            layers.Conv1D(filters=latent_dim, kernel_size=3, activation=None, padding="same", name="encoder_output_pre_vq"),
            layers.BatchNormalization(),
       ], name='encoder')
            

        # VQ Layer
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost, name="vector_quantizer")

       
        
        self.decoder = tf.keras.Sequential([
           
            layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, activation="relu", padding="same"),
            
            layers.Conv1DTranspose(filters=64, kernel_size=4, strides=2, activation="relu", padding="same"),
         
            layers.Conv1D(filters=input_shape[-1], kernel_size=8, activation=None, padding="same")

        ], name='decoder')

    def call(self, x):
        encoder_output = self.encoder(x)
        quantized_latents = self.vq_layer(encoder_output)
        reconstructions = self.decoder(quantized_latents)
        
        # Add the reconstruction loss
        reconstruction_loss = tf.reduce_mean((x - reconstructions)**2, name="reconstruction_loss")
        #self.add_loss(reconstruction_loss)
        
        return reconstructions

    def get_latent_representation(self, x):
        # This method can be used to get the actual quantized embeddings for visualization
        encoder_output = self.encoder(x)
        
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
def train_and_evaluate_vqvae(train_data, test_data, input_shape, latent_dim, num_embeddings, commitment_cost, epochs=10, batch_size=32,learning_rate=1e-4):
    # Defaults angegeben, damit es beim Debugging nicht zu Problemen kommt. Defaults werden im main überschrieben
    # Instanziieren des VQ-VAE Modells
    vq_vae = VQVAE(input_shape, latent_dim, num_embeddings, commitment_cost)
  


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    vq_vae.compile(optimizer=optimizer, loss = 'mse') # Keras will automatically pick up losses added via add_loss() , mse=mean sqaured error
    
    print(f"\nStarte Training:")
    print(f"  Latent Dim: {latent_dim}")
    print(f"  Codebook Size: {num_embeddings}")
    print(f"  β (Commitment Cost): {commitment_cost}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Epochs: {epochs}, Batch Size: {batch_size}")
    
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
    

    return vq_vae, history, total_loss # Oder history, reconstruction_mse
