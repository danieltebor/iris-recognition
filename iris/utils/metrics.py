import numpy as np
from sklearn.metrics import pairwise_distances, roc_curve


def compute_inter_and_intra_class_distances(
    embeddings_x: np.ndarray,
    labels_x: np.ndarray,
    embeddings_y: np.ndarray = None,
    labels_y: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    assert len(embeddings_x) == len(labels_x)
    if embeddings_y is not None:
        assert labels_y is not None
        assert len(embeddings_y) == len(labels_y)
    
    distances = pairwise_distances(
        X=embeddings_x,
        Y=embeddings_y,
        metric='euclidean',
        n_jobs=-1,
    )
    intraclass_distances = []
    interclass_distances = []
    
    if embeddings_y is None:
        for row in range(len(labels_x)):
            row_label = labels_x[row]
            for col in range(row+1, len(labels_x)): # Upper triangular matrix.
                col_label = labels_x[col]
                if row_label == col_label:
                    intraclass_distances.append(distances[row, col])
                else:
                    interclass_distances.append(distances[row, col])
    else:
        for row in range(len(labels_x)):
            row_label = labels_x[row]
            for col in range(len(labels_y)):
                col_label = labels_y[col]
                if row_label == col_label:
                    intraclass_distances.append(distances[row, col])
                else:
                    interclass_distances.append(distances[row, col])
                
    return np.array(intraclass_distances), np.array(interclass_distances)

def compute_distance_roc(embeddings_x: np.ndarray,
                         labels_x: np.ndarray,
                         embeddings_y: np.ndarray = None,
                         labels_y: np.ndarray = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    intraclass_distances, interclass_distances = compute_inter_and_intra_class_distances(embeddings_x, labels_x, embeddings_y, labels_y)
    distances = -np.concatenate([intraclass_distances, interclass_distances])
    labels = [1]*len(intraclass_distances) + [0]*len(interclass_distances)
    
    return roc_curve(labels, distances)