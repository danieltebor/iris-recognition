import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('Agg')


def intra_inter_class_distance_histogram(intra_distances: np.ndarray,
                                         inter_distances: np.ndarray,
                                         save_dir: str,
                                         filename: str = 'intra-inter-class-distance-hist',
                                         title: str = None,
                                         legend_title = None):
    plt.figure()
    plt.hist(intra_distances, bins=100, alpha=0.5, color='blue', label='Intra-class Distances', density=True)
    plt.hist(inter_distances, bins=100, alpha=0.5, color='red', label='Inter-class Distances', density=True)
    if title is not None:
        plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend(title=legend_title, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def roc(results: dict,
        save_dir: str,
        filename: str = 'roc',
        title: str = None,
        legend_title: str = None,
        plot_eer: bool = True,
        include_auc: bool = True,
        x_lim: tuple = (0, 1),
        y_lim: tuple = (0, 1)):
    plt.figure()
    
    colors = plt.cm.get_cmap('tab20', len(results)).colors
    for key, result, color in zip(results.keys(), results.values(), colors):
        fpr = result['fpr']
        tpr = result['tpr']
        
        label = key
        if plot_eer:
            eer = result['eer']
            label += f' (EER={eer:.4f})'
        if include_auc:
            auc = result['auc']
            label += f' (AUC={auc:.4f})'
        
        plt.plot(fpr, tpr, label=label, color=color)
        
        if plot_eer:
            eer = result['eer']
            plt.scatter(eer, 1 - eer, color=color)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    if plot_eer:
        plt.plot([0, 1], [1, 0], linestyle='--', color='black')
    if title is not None:
        plt.title(title)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.legend(title=legend_title, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def tsne_2d(embeddings_2d: np.ndarray,
            labels: np.ndarray,
            save_dir: str,
            filename: str = 'tsne',
            title: str = None,
            legend_title: str = None):
    plt.figure()
    
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20', len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[cmap(i)],
            label=label
        )
    
    if title is not None:
        plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    if len(unique_labels) <= 10:
        plt.legend(title=legend_title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def tsne_3d(embeddings_3d: np.ndarray,
            labels: np.ndarray,
            save_dir: str,
            filename: str = 'tsne',
            title: str = None,
            legend_title: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20', len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embeddings_3d[mask, 0],
            embeddings_3d[mask, 1],
            embeddings_3d[mask, 2],
            c=[cmap(i)],
            label=label,
            alpha=0.5,
        )
        
    if title is not None:
        plt.title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    if len(unique_labels) <= 10:
        plt.legend(title=legend_title)
    plt.tight_layout()
    ax.view_init(elev=30, azim=0)
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()