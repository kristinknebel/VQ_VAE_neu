# src/visualization.pyparams_info = params_str

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.manifold import TSNE
import umap
import numpy as np
import tensorflow as tf # TensorFlow ist jetzt hier notwendig für tf.gather und tf.reshape
from typing import Optional, Dict, Any, Tuple, Iterable
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
    pio.write_html(fig, file=filename, include_plotlyjs="cdn", auto_open=False)
    print(f"Latent-Space-Visualisierung gespeichert unter: {filename}")


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

# === Codebook-Analyse für VQ-VAE (TensorFlow/Keras) ===

def _find_encoder(model: tf.keras.Model) -> tf.keras.Model:
    # 1) bevorzugt: Attribut
    if hasattr(model, "encoder") and isinstance(model.encoder, tf.keras.Model):
        return model.encoder
    # 2) Try by name
    try:
        return model.get_layer("encoder")
    except Exception:
        pass
    # 3) Fallback: komplette Vorwärtsrechnung nutzen und vor dem VQ layer tap-in (nicht trivial)
    raise AttributeError(
        "Konnte keinen Encoder finden. Erwarte model.encoder oder eine Layer mit Namen 'encoder'."
    )

def _find_vq_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    # 1) direkte Attribute, die oft benutzt werden
    for attr in ("vq_layer", "quantizer", "vq", "vector_quantizer"):
        if hasattr(model, attr):
            return getattr(model, attr)
    # 2) nach Layer-Typname/-Name suchen
    for layer in model.layers:
        name = layer.name.lower()
        cls = layer.__class__.__name__.lower()
        if "vectorquantizer" in cls or "vectorquantizer" in name or "vq" in name:
            return layer
    # 3) tiefer in Submodels schauen
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                return _find_vq_layer(layer)
            except Exception:
                continue
    raise AttributeError("Konnte keinen VQ-Layer finden (z. B. 'vq_layer', 'VectorQuantizer').")

def _get_codebook_matrix(vq_layer: tf.keras.layers.Layer) -> tf.Tensor:
    """
    Versucht, die Codebook-Matrix (K x D) zu extrahieren.
    Häufige Namen: 'embedding', 'embeddings', 'codebook'.
    """
    # Keras-Layer.weights: Liste tf.Variable; häufig Name enthält 'embedding'
    cand_vars = []
    for w in vq_layer.weights:
        n = w.name.lower()
        if any(k in n for k in ("embedding", "embeddings", "codebook")) and len(w.shape) == 2:
            cand_vars.append(w)
    if cand_vars:
        # wähle die erste 2D-Matrix
        return tf.convert_to_tensor(cand_vars[0])
    # Manche Implementationen halten das Codebook als Attribut
    for attr in ("embedding", "embeddings", "codebook", "codebook_embedding"):
        if hasattr(vq_layer, attr):
            var = getattr(vq_layer, attr)
            if isinstance(var, (tf.Variable, tf.Tensor)) and len(var.shape) == 2:
                return tf.convert_to_tensor(var)
    raise AttributeError("Codebook-Matrix (KxD) nicht gefunden.")

def _vq_forward_indices(vq_layer, z_e: tf.Tensor) -> Optional[tf.Tensor]:
    """
    Ruft den VQ-Layer auf und versucht, Indizes zu erhalten.
    Erwartete Rückgaben (je nach Implementierung):
      - (z_q, indices) oder (z_q, indices, aux)
      - dict mit Schlüssel 'indices' oder 'encoding_indices'
      - nur z_q (dann None → später selbst berechnen)
    """
    out = vq_layer(z_e, training=False)
    # Tupel-Varianten
    if isinstance(out, (tuple, list)):
        # finde erstes Tensor-Ähnliche mit ganzzahligem dtype als Indices
        for item in out[1:]:
            if tf.is_tensor(item) and item.dtype.is_integer:
                return tf.cast(item, tf.int32)
        # manchmal ist ein dict im Tupel
        for item in out:
            if isinstance(item, dict):
                for k in ("indices", "encoding_indices", "codes"):
                    if k in item:
                        t = item[k]
                        if tf.is_tensor(t):
                            return tf.cast(t, tf.int32)
        return None
    # Dict-Variante
    if isinstance(out, dict):
        for k in ("indices", "encoding_indices", "codes"):
            if k in out and tf.is_tensor(out[k]):
                return tf.cast(out[k], tf.int32)
        return None
    # Nur Tensor zurück → keine Indices
    return None

def _nearest_code_indices_manual(E: tf.Tensor, z_e: tf.Tensor, chunk: int = 32768) -> tf.Tensor:
    """
    Fallback: berechnet Indizes manuell über nächste Codebook-Vektoren.
    E: (K, D), z_e: (..., D) -> flach zu (N, D)
    Chunking, um Speicher zu sparen.
    """
    z = tf.reshape(z_e, [-1, tf.shape(z_e)[-1]])  # (N, D)
    K = tf.shape(E)[0]
    N = tf.shape(z)[0]

    idx_all = []
    start = tf.constant(0)
    while True:
        end = tf.minimum(start + chunk, N)
        z_chunk = z[start:end]                   # (n, D)
        # d^2 = ||z||^2 + ||E||^2 - 2 z E^T
        zz = tf.reduce_sum(tf.square(z_chunk), axis=1, keepdims=True)      # (n,1)
        EE = tf.reduce_sum(tf.square(E), axis=1, keepdims=True)            # (K,1)
        # (n,K)
        d2 = zz + tf.transpose(EE) - 2.0 * tf.linalg.matmul(z_chunk, E, transpose_b=True)
        idx = tf.argmin(d2, axis=1, output_type=tf.int32)                  # (n,)
        idx_all.append(idx)
        if tf.equal(end, N):
            break
        start = end
    return tf.concat(idx_all, axis=0)  # (N,)

def analyze_codebook_usage(
    model: tf.keras.Model,
    dataset: Iterable,
    max_batches: Optional[int] = 50,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Analysiert, wie viele Codebook-Einträge genutzt werden und wie gleichmäßig.
    - model: dein VQ-VAE Keras-Modell (mit model.encoder und einem VQ-Layer)
    - dataset: tf.data.Dataset oder beliebiger (x, y)-Iterator
    - max_batches: Anzahl ausgewerteter Batches (None = alle)
    - plot: Balkendiagramm der Häufigkeiten zeichnen

    Rückgabe:
      {
        'unique_indices': np.ndarray shape (U,),             # verwendete Indizes
        'counts': np.ndarray shape (U,),                     # Häufigkeiten pro Index
        'utilization': float,                                # U / K
        'num_embeddings': int,                               # K
        'total_assignments': int,                            # Summe counts
      }
    """
    enc = _find_encoder(model)
    vq = _find_vq_layer(model)
    E = _get_codebook_matrix(vq)            # (K, D)
    K = int(E.shape[0])

    used_indices = []

    @tf.function(reduce_retracing=True)
    def _enc_call(x):
        return enc(x, training=False)

    it = iter(dataset)
    b = 0
    while True:
        if max_batches is not None and b >= max_batches:
            break
        try:
            batch = next(it)
        except StopIteration:
            break
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        z_e = _enc_call(tf.convert_to_tensor(x))

        # 1) Versuche, Indizes direkt vom VQ-Layer zu bekommen
        idx = _vq_forward_indices(vq, z_e)
        if idx is None:
            # 2) Sonst manuell via nächstem Codebook-Vektor
            idx = _nearest_code_indices_manual(E, z_e)

        idx_np = idx.numpy().ravel()
        used_indices.append(idx_np)
        b += 1

    if not used_indices:
        raise RuntimeError("Dataset lieferte keine Batches oder Batches sind leer.")

    indices = np.concatenate(used_indices, axis=0)
    unique, counts = np.unique(indices, return_counts=True)
    utilization = float(len(unique)) / float(K)
    total = int(counts.sum())

    if plot:
        plt.figure(figsize=(10, 4))
        plt.bar(unique, counts)
        plt.xlabel("Codebook-Index")
        plt.ylabel("Häufigkeit")
        plt.title(f"Codebook-Nutzung: {len(unique)}/{K} aktiv ({utilization*100:.1f}%)")
        plt.tight_layout()
        plt.show()

    return {
        "unique_indices": unique,
        "counts": counts,
        "utilization": utilization,
        "num_embeddings": K,
        "total_assignments": total,
    }
