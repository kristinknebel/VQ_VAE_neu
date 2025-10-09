# src/visualization.pyparams_info = params_str

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap
import numpy as np
import tensorflow as tf # TensorFlow ist jetzt hier notwendig für tf.gather und tf.reshape
from .config import CATEGORY_COLORS # Sicherstellen, dass dies korrekt ist, ggf. direkt die Farben definieren
# from .models import VQVAE # Importieren Sie Ihre VQVAE-Klasse, falls Sie sie für Typ-Hints benötigen.
#                         # In der Regel übergeben Sie eine Instanz, so dass der direkte Import nicht zwingend ist.


# ---- Funktion zur Visualisierung des Latenzraums ----
# Diese Funktion nimmt jetzt ein VQVAE-Modell an, nicht nur einen Encoder
def visualize_latent_space(vq_vae_model, data, labels, n_components=3, method='TSNE', filename="", params_info=""):
    """
    Visualisiert den gelernten, quantisierten Latent-Raum eines VQ-VAE.
    Es werden die den Eingabedaten zugeordneten, quantisierten Codebook-Vektoren visualisiert.

    Args:
        vq_vae_model: Die trainierte Instanz Ihres VQVAE-Modells.
        data (np.array): Die EKG-Snippets, deren Latent-Repräsentationen visualisiert werden sollen.
                         Shape: (num_samples, snippet_length, num_channels)
        labels (np.array): Die entsprechenden Labels für die EKG-Snippets.
        n_components (int): Anzahl der Komponenten für die Dimensionsreduktion (2 für 2D, 3 für 3D).
        method (str): Methode zur Dimensionsreduktion ('TSNE' oder 'UMAP').
    """
    if not isinstance(vq_vae_model, tf.keras.Model) or not hasattr(vq_vae_model, 'encoder') or not hasattr(vq_vae_model, 'vq_layer'):
        raise ValueError("vq_vae_model muss eine Instanz von VQVAE mit 'encoder' und 'vq_layer' Attributen sein.")

    print(f"Starte Visualisierung des Latent-Raums mit {method}...")

    # Schritt 1: Encoder-Output vor der Quantisierung erhalten
    # Der Encoder gibt Features aus (batch_size, seq_len_reduced, latent_dim)
    encoded_features = vq_vae_model.encoder.predict(data)
    
    # Die latente Dimension ist die letzte Dimension des Encoder-Outputs (embedding_dim im VQ-Layer)
    latent_dim = encoded_features.shape[-1]
    
    # Schritt 2: Flache Repräsentation des Encoder-Outputs erstellen
    # Notwendig für die `get_code_indices` Methode des VectorQuantizer,
    # da diese typischerweise einen 2D-Tensor (num_vectors_to_quantize, embedding_dim) erwartet.
    # num_vectors_to_quantize = batch_size * seq_len_reduced
    num_samples_flat = encoded_features.shape[0] * encoded_features.shape[1]
    flat_encoded_features = encoded_features.reshape(num_samples_flat, latent_dim)

    # Schritt 3: Indizes der nächstgelegenen Codevektoren aus dem Codebook abrufen
    # Die `get_code_indices`-Methode Ihrer `VectorQuantizer`-Klasse.
    # Sie gibt einen 1D-Tensor von Indizes zurück.
    code_indices = vq_vae_model.vq_layer.get_code_indices(flat_encoded_features)
    
    # Schritt 4: Die tatsächlichen quantisierten Embeddings aus dem Codebook abrufen
    # `vq_vae_model.vq_layer.embeddings` ist der trainierbare Tensor des Codebooks.
    # `tf.gather` verwendet die Indizes, um die entsprechenden Vektoren zu extrahieren.
    quantized_latents_flat = tf.gather(vq_vae_model.vq_layer.embeddings, code_indices)
    
    # Schritt 5: Aggregation der quantisierten Embeddings pro EKG-Snippet
    # Da jedes EKG-Snippet in eine Sequenz von latenten Vektoren kodiert wird (seq_len_reduced),
    # müssen wir diese zu einer einzelnen Repräsentation pro Snippet zusammenfassen.
    # Der Mittelwert ist eine gängige Methode.
    
    # Zuerst zurückformen zu (batch_size, seq_len_reduced, latent_dim)
    quantized_latents_reshaped = quantized_latents_flat.numpy().reshape(encoded_features.shape)
    
    # Dann den Mittelwert über die Sequenzlänge (Achse 1) bilden
    latent_representations_for_tsne = np.mean(quantized_latents_reshaped, axis=1) # Shape: (batch_size, latent_dim)
    
    num_samples_for_tsne = latent_representations_for_tsne.shape[0]
    
    # Überprüfung für TSNE perplexity
    perplexity_val = min(30, num_samples_for_tsne - 1)
    if num_samples_for_tsne <= 1: # TSNE requires at least 2 samples
        print("Nicht genügend Samples für T-SNE/UMAP Visualisierung.")
        return

    # Dimensionsreduktion
    if method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(latent_representations_for_tsne)
        title = f'UMAP Visualisierung des quantisierten Latentspace (Single Label)'
    else: # Default to TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val)
        reduced_embeddings = reducer.fit_transform(latent_representations_for_tsne)
        title = f'TSNE Visualisierung des quantisierten Latentspace (Single Label),{params_info}'
    

    # Die restliche Plot-Logik bleibt gleich
    # Sicherstellen, dass CATEGORY_COLORS korrekt aus config importiert wird oder hier definiert ist
    category_colors = {"NORM": '#1f77b4' , "LVH": '#ff7f0e', "CLBBB": '#2ca02c', "CRBBB": '#d62728', "IRBBB": '#9467bd', "LAFB": '#8c564b ', "WPW": '#e377c2', "1AVB": '#7f7f7f', "AFIB": '#17becf', "AFLT": '#aec7e8', "IVCD": '#bcbd22'} # Updated IVCD
    
    # Sicherstellen, dass alle Labels in category_colors vorhanden sind
    # Oder einen Fallback für unbekannte Labels definieren
    point_colors = [category_colors.get(label, '#000000') for label in labels] # Fallback to black for unknown labels

    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        mode='markers',  # Zeigt Punkte an
        marker=dict(
            size=8,
            color=point_colors  
        ),
        text=labels,       # Text, der beim Hovern angezeigt wird (optional)
        hoverinfo='text'
    )])

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        legend=dict(
            title="SCP-Label",
            itemsizing='constant',
            orientation='v',
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.update_layout(layout)
    fig.write_html(filename) 
    fig.show()

# ---- Funktion zur Visualisierung der EKG-Rekonstruktionen ----
# Diese Funktion bleibt gleich, da sie nur Original und Rekonstruktion plottet
def plot_ecg_reconstructions(original_ecg_snippets, reconstructed_ecg_snippets, num_examples=5, filename="", params_info=""):
    """
    Plottet eine Vergleichsansicht von originalen und rekonstruierten EKG-Snippets.

    Args:
        original_ecg_snippets (np.array): Die ursprünglichen EKG-Snippets.
        reconstructed_ecg_snippets (np.array): Die vom Autoencoder rekonstruierten EKG-Snippets.
        num_examples (int): Anzahl der Beispiele, die geplottet werden sollen.
        filename (str): Dateiname für die Speicherung des Plots.
    """
    # Sicherstellen, dass die Anzahl der Kanäle korrekt ist (z.B. 12 für PTB-XL)
    # Und dass die Snippets die Form (num_samples, snippet_length, num_channels) haben
    
    total_channels = original_ecg_snippets.shape[2] # Annahme: letzter Dim ist Kanäle
    snippet_length = original_ecg_snippets.shape[1] # Annahme: mittlere Dim ist Länge

    # Erstelle Subplots: num_examples Zeilen, 2*total_channels Spalten (Original + Rekonstruktion pro Kanal)
    fig, axes = plt.subplots(num_examples, total_channels * 2, figsize=(20, num_examples * 2), squeeze=False)

    for i in range(num_examples):
        original_snippet = original_ecg_snippets[i]
        reconstructed_snippet = reconstructed_ecg_snippets[i]

        for channel in range(total_channels):
            # Original EKG
            ax_orig = axes[i, channel * 2]
            ax_orig.plot(original_snippet[:, channel])
            if i == 0: # Titel nur für die erste Zeile
                ax_orig.set_title(f'Orig. Ch {channel+1}', fontsize=8)
            ax_orig.set_xticks([])
            ax_orig.set_yticks([])
            
            # Rekonstruiertes EKG
            ax_recon = axes[i, channel * 2 + 1]
            ax_recon.plot(reconstructed_snippet[:, channel])
            if i == 0: # Titel nur für die erste Zeile
                ax_recon.set_title(f'Rec. Ch {channel+1}', fontsize=8)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---- Neue Funktion zur Visualisierung der Codebook-Embeddings ----
# Diese ist sehr wichtig für VQ-VAE, um die gelernten diskreten Vektoren zu verstehen.
def visualize_codebook_embeddings(vq_layer, n_components=3, method='TSNE', filename="", params_info=""):
    """
    Visualisiert die gelernten Codebook-Embeddings (die diskreten Vektoren).

    Args:
        vq_layer: Die Instanz Ihrer VectorQuantizer-Schicht (z.B. vq_vae_model.vq_layer).
        n_components (int): Anzahl der Komponenten für die Dimensionsreduktion (2 für 2D, 3 für 3D).
        method (str): Methode zur Dimensionsreduktion ('TSNE' oder 'UMAP').
        filename (str): Dateiname für die Speicherung des Plots.
    """
    if not hasattr(vq_layer, 'embeddings'):
        raise ValueError("vq_layer muss ein 'embeddings' Attribut haben (der Codebook-Tensor).")

    # Zugriff auf den trainierbaren Tensor der Embeddings
    embeddings = vq_layer.embeddings.numpy() # Convert to NumPy array
    
    num_embeddings = embeddings.shape[0]
    embedding_dim = embeddings.shape[1]
    
    print(f"Visualisiere Codebook mit {num_embeddings} Embeddings, Dimension {embedding_dim} mit {method}...")

    if num_embeddings <= 1:
        print("Nicht genügend Embeddings für T-SNE/UMAP Visualisierung.")
        return

    # Reduktion der Dimensionen für die Visualisierung
    perplexity_val = min(30, num_embeddings - 1)
    if method == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = f'UMAP Visualisierung der Codebook Embeddings'
    else: # Default to TSNE
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity_val)
        reduced_embeddings = reducer.fit_transform(embeddings)
        title = f'TSNE Visualisierung der Codebook Embeddings,{params_info}'

    # Erstellung des interaktiven 3D-Plots mit Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        z=reduced_embeddings[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='blue', # Alle Codebook-Vektoren in einer Farbe
            opacity=0.8
        ),
        text=[f'Code {i}' for i in range(num_embeddings)], # Text für Hover
        hoverinfo='text'
    )])

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.update_layout(layout)
    fig.write_html(filename)
