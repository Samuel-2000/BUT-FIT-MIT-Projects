# audio_gmm.py
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import torchaudio
import torch
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

# ----------------------
# Core Components
# ----------------------

def get_model_filename(model_type: str, augment: bool) -> str:
    aug_suffix = 'augmented' if augment else 'original'
    return f"{model_type}_{aug_suffix}.pkl"

class AudioDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory {self.root} does not exist")
            
        self.files = list(self.root.rglob("*.wav"))
        if not self.files:
            raise ValueError(f"No WAV files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(str(file_path))
        return {
            "name": str(file_path),
            "data": waveform,
            "sample_rate": sample_rate
        }


class Pipeline:
    def __init__(self, in_freq: int = 16000, out_freq: int = 16000,
                 n_fft: int = 256, win_length: int = 200, win_hop: int = 100,
                 n_mels: int = 23, n_mfcc: int = 13):
        self.resample = torchaudio.transforms.Resample(in_freq, out_freq)
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=out_freq,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": win_hop,
                "win_length": win_length
            }
        )

    def apply(self, x: Dict[str, Any]) -> np.ndarray:
        try:
            if x["sample_rate"] != self.resample.new_freq:
                x["data"] = self.resample(x["data"])
            
            mfcc = self.mfcc(x["data"]).squeeze(0)
            delta = torchaudio.functional.compute_deltas(mfcc.unsqueeze(0)).squeeze(0)
            ddelta = torchaudio.functional.compute_deltas(delta.unsqueeze(0)).squeeze(0)
            
            features = torch.cat([mfcc, delta, ddelta], dim=0).T.numpy()
            
            stats = np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0),
                np.percentile(features, 75, axis=0),
                np.percentile(features, 25, axis=0)
            ])
            
            return stats
        except Exception as e:
            print(f"Error processing {x['name']}: {str(e)}")
            raise

class AudioAugmenter:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.noise_std = 0.005
        self.vol_gain_db = [-10, 10]

    def apply_random_augment(self, waveform):
        aug_type = np.random.choice(['noise', 'vol'], p=[0.5, 0.5])

        if aug_type == 'noise':
            return waveform + torch.randn_like(waveform) * self.noise_std
        elif aug_type == 'vol':
            gain = np.random.uniform(*self.vol_gain_db)
            return torchaudio.functional.gain(waveform, gain)
        
        return waveform


# ----------------------
# Feature Loading
# ----------------------

def load_features(
    dataset_path: str,
    pipeline: Pipeline,
    augment: bool = False,
    num_augments: int = 5,
    parallel: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        dataset = AudioDataset(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Dataset error: {str(e)}")

    all_features = []
    all_labels = []
    augmenter = AudioAugmenter(pipeline.resample.new_freq) if augment else None

    def process_item(item):
        try:
            features = pipeline.apply(item)
            speaker_id = get_true_label(item["name"])
            
            if augment:
                for _ in range(num_augments):
                    aug_waveform = augmenter.apply_random_augment(item["data"])
                    augmented_item = {
                        "data": aug_waveform,
                        "sample_rate": item["sample_rate"],
                        "name": item["name"]
                    }
                    aug_features = pipeline.apply(augmented_item)
                    yield aug_features.reshape(1, -1), np.array([speaker_id])
            
            yield features.reshape(1, -1), np.array([speaker_id])
        except Exception as e:
            print(f"Skipping {item['name']}: {str(e)}")
            return None

    if parallel:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for result in executor.map(process_item, dataset):
                if result:
                    for feat, lbl in result:
                        all_features.append(feat)
                        all_labels.append(lbl)
    else:
        for item in dataset:
            for result in process_item(item):
                if result:
                    feat, lbl = result
                    all_features.append(feat)
                    all_labels.append(lbl)

    features_arr = np.concatenate(all_features, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    
    return features_arr, labels_arr




# ----------------------
# Model Training
# ----------------------
    
def train_and_evaluate_gmm(train_folder: str, test_folder: str, augment: bool = False, force_train: bool = False) -> List[Dict[str, Any]]:
    model_filename = get_model_filename('gmm', augment)

    pipeline = Pipeline()

    
    if not force_train and os.path.exists(model_filename):
        model_data = joblib.load(model_filename)
        gmms = model_data['gmms']
        scaler = model_data['scaler']
    else:
        try:
            train_features, train_labels = load_features(
                train_folder, pipeline, augment=augment, parallel=True
            )
        except ValueError as e:
            raise SystemExit(f"Feature extraction failed: {str(e)}")

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)

        gmms = {}
        unique_speakers = np.unique(train_labels)
        for speaker_id in unique_speakers:
            speaker_mask = (train_labels == speaker_id)
            speaker_features = train_features_scaled[speaker_mask]
            
            if len(speaker_features) == 0:
                print(f"No features found for speaker {speaker_id}, skipping.")
                continue
                
            n_components = 1 # best accuracy with 1, although cant be called Mixture model anymore.
            
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                reg_covar=1e-4,
                max_iter=2000,
                random_state=0
            ).fit(speaker_features)
            gmms[speaker_id] = gmm

        model_data = {
            'gmms': gmms,
            'scaler': scaler,
        }
        joblib.dump(model_data, model_filename)
        
        plot_gmm_components(
            gmms,
            train_features_scaled,
            train_labels,
            'gmm_components_multiple.png'
        )
        plot_feature_space(train_features_scaled, train_labels, 'gmm_features.png')
    

    test_data = []
    test_dataset = AudioDataset(test_folder)
    for item in test_dataset:
        try:
            raw_features = pipeline.apply(item)
            scaled_features = scaler.transform(raw_features.reshape(1, -1))
            test_data.append((item["name"], scaled_features))
        except Exception as e:
            print(f"Skipping {item['name']}: {e}")


    results = []
    sorted_speakers = sorted(gmms.keys())
    for name, test_features in test_data:
        scores = []
        for speaker_id in sorted_speakers:
            log_likelihood = gmms[speaker_id].score_samples(test_features)
            scores.append(np.mean(log_likelihood))
        predicted = sorted_speakers[np.argmax(scores)]
        results.append({
            "name": name,
            "scores": scores,
            "predicted": predicted
        })
        
    
    return results





def train_kmeans(features: np.ndarray, labels: np.ndarray) -> Tuple[KMeans, dict]:
    # Determine number of clusters from unique speakers
    unique_speakers = np.unique(labels)
    n_clusters = len(unique_speakers)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=20,
        algorithm='elkan',
        random_state=0
    ).fit(features)
    
    # Create cluster to speaker mapping
    cluster_map = {}
    for cluster_id in range(n_clusters):
        # Get indices of points in this cluster
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        
        # Find most common speaker in cluster
        cluster_speakers = labels[cluster_indices]
        unique, counts = np.unique(cluster_speakers, return_counts=True)
        mapped_speaker = unique[np.argmax(counts)]
        
        cluster_map[cluster_id] = mapped_speaker
    
    return kmeans, cluster_map





def train_and_evaluate_kmeans(train_folder: str, test_folder: str, augment: bool = False, force_train: bool = False) -> List[Dict[str, Any]]:
    pipeline = Pipeline()
    
    try:
        features, labels = load_features(
            train_folder,
            pipeline,
            augment=augment,
            parallel=True
        )
    except ValueError as e:
        raise SystemExit(f"Feature extraction failed: {str(e)}")

    scaler = StandardScaler()
    try:
        features_scaled = scaler.fit_transform(features)
    except ValueError as e:
        raise SystemExit(f"Scaling failed: {str(e)}")

    model_filename = get_model_filename('kmeans', augment)
    if not force_train and os.path.exists(model_filename):
        model_data = joblib.load(model_filename)
        kmeans = model_data['kmeans']
        cluster_map = model_data['cluster_map']
        
    else:
        try:
            kmeans, cluster_map = train_kmeans(features_scaled, labels)
        except ValueError as e:
            raise SystemExit(f"KMeans training failed: {str(e)}")

        # Generate visualizations
        plot_kmeans_components(kmeans, features_scaled, labels, 'kmeans_cluster_analysis.png')

        model_data = {
            'kmeans': kmeans,
            'cluster_map': cluster_map,
        }
        joblib.dump(model_data, model_filename)


    test_data = []
    try:
        test_dataset = AudioDataset(test_folder)
        for item in test_dataset:
            try:
                feat = scaler.transform(pipeline.apply(item).reshape(1, -1))
                test_data.append((item["name"], feat.squeeze()))
            except Exception as e:
                print(f"Skipping {item['name']}: {e}")
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"Test dataset error: {str(e)}")
    
    return [{
        "name": name,
        "predicted": cluster_map[kmeans.predict(feat.reshape(1, -1))[0]],
        "scores": ['NaN'] * kmeans.n_clusters  # Replace scores with 'NaN' strings
    } for (name, feat) in test_data]






# ----------------------
# Visualization & Output
# ----------------------


def get_true_label(filename: str) -> str:
    """Extract speaker ID from parent directory name"""
    path = Path(filename)
    return path.parent.name


def print_predictions(pred: List[Dict[str, Any]]) -> None:
    #correct = 0
    #total = 0
    for p in pred:
    #   true_label = get_true_label(p['name'])
        predicted = p['predicted']
        # Handle both numerical scores and 'NaN' strings
        if isinstance(p['scores'][0], str):
            scores_str = ' '.join(p['scores'])
        else:
            scores_str = ' '.join([f"{s:.2f}" for s in p['scores']])
        print(f"{Path(p['name']).stem} {predicted} {scores_str}")
    #    if predicted == true_label:
    #        correct += 1
    #    total += 1
    #print(f"\nAccuracy: {correct/total:.2%} ({correct}/{total})")


# Create directory for plots
os.makedirs("plots", exist_ok=True)


def plot_feature_space(features: np.ndarray, labels: np.ndarray, filename: str) -> None:
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Convert string labels to numerical indices
    unique_labels = np.unique(labels)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[lbl] for lbl in labels])

    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=numeric_labels, cmap='tab20', 
                         alpha=0.6, edgecolor='k')
    
    # Create custom colorbar with actual labels
    cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
    cbar.ax.set_yticklabels(unique_labels)
    
    plt.title("Speaker Feature Space")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"plots/{filename}", bbox_inches='tight')
    plt.close()

def plot_gmm_components(gmms: Dict[str, GaussianMixture], 
                       features: np.ndarray,
                       labels: np.ndarray,
                       filename: str) -> None:
    """Visualize GMM components with folder-based IDs"""
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    plt.figure(figsize=(15, 10))
    cmap = plt.get_cmap('tab20')
    unique_speakers = np.unique(labels)
    
    # Create label to index mapping
    label_map = {label: idx for idx, label in enumerate(unique_speakers)}

    # Plot all data points
    for idx, speaker_id in enumerate(unique_speakers):
        mask = np.array(labels) == speaker_id
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    color=cmap(idx/len(unique_speakers)),
                    alpha=0.3, edgecolor='k',
                    label=speaker_id)

    # Plot GMM components
    for idx, (speaker_id, gmm) in enumerate(gmms.items()):
        color = cmap(label_map[speaker_id]/len(unique_speakers))
        
        # Transform GMM means to PCA space
        means_2d = pca.transform(gmm.means_)
        
        # Plot means as stars
        plt.scatter(means_2d[:, 0], means_2d[:, 1], s=150,
                   marker='*', color=color, edgecolor='white',
                   linewidth=1, label=f'{speaker_id} GMM')

        # Plot covariance ellipses
        for mean, cov in zip(gmm.means_, gmm.covariances_):
            if gmm.covariance_type == 'diag':
                cov_matrix = np.diag(cov)
            else:
                cov_matrix = cov
                
            cov_2d = pca.components_ @ cov_matrix @ pca.components_.T
            v, w = np.linalg.eigh(cov_2d)
            angle = np.degrees(np.arctan2(w[0][1], w[0][0]))
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            
            ell = plt.matplotlib.patches.Ellipse(
                pca.transform(mean.reshape(1, -1))[0],
                v[0], v[1], angle=angle,
                color=color, alpha=0.2, linewidth=2
            )
            plt.gca().add_artist(ell)

    plt.title("GMM Components by Speaker")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig(f"plots/{filename}", bbox_inches='tight')
    plt.close()

# ----------------------
# KMeans Visualization Update
# ----------------------

def plot_kmeans_components(kmeans: KMeans, features: np.ndarray, 
                          labels: np.ndarray, filename: str):
    """Updated for folder-based speaker IDs"""
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    unique_speakers = np.unique(labels)
    
    plt.figure(figsize=(18, 8))
    cmap = plt.get_cmap('tab20')

    # Plot clusters
    plt.subplot(131)
    for cluster_id in range(kmeans.n_clusters):
        cluster_mask = kmeans.labels_ == cluster_id
        plt.scatter(features_2d[cluster_mask, 0], features_2d[cluster_mask, 1],
                    color=cmap(cluster_id/kmeans.n_clusters),
                    alpha=0.6, edgecolor='k',
                    label=f'Cluster {cluster_id}')
    plt.title("KMeans Clusters")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    # Plot true speaker distribution
    plt.subplot(132)
    for idx, speaker_id in enumerate(unique_speakers):
        speaker_mask = np.array(labels) == speaker_id
        plt.scatter(features_2d[speaker_mask, 0], features_2d[speaker_mask, 1],
                    color=cmap(idx/len(unique_speakers)),
                    alpha=0.6, edgecolor='k',
                    label=speaker_id)
    plt.title("True Speakers")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    # Plot cluster centers
    plt.subplot(133)
    centers_2d = pca.transform(kmeans.cluster_centers_)
    for cluster_id in range(kmeans.n_clusters):
        plt.scatter(centers_2d[cluster_id, 0], centers_2d[cluster_id, 1],
                    color=cmap(cluster_id/kmeans.n_clusters),
                    marker='X', s=200, edgecolor='k',
                    label=f'Cluster {cluster_id} Center')
    plt.title("Cluster Centers")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    plt.tight_layout()
    plt.savefig(f"plots/{filename}", bbox_inches='tight')
    plt.close()