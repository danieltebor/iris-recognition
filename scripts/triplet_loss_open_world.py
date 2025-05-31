import copy
from datetime import datetime
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_optimizer import AdamP
from torch.nn import TripletMarginLoss
from torch.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import auc
from safetensors.torch import save_file, load_file

from iris.common.constants import OUT_DIR
from iris.datasets.iris_dataset import IrisDataset, FRONTAL_ANGLE, IRIS_IMAGE_ANGLES
from iris.datasets.iris_triplet_dataset import IrisTripletDataset
from iris.models.model_factory import ModelConfig, ModelFactory
from iris.utils.metrics import compute_inter_and_intra_class_distances, compute_distance_roc
from iris.utils.plotting import intra_inter_class_distance_histogram, roc, tsne_2d, tsne_3d


def train_model(config: dict) -> str:
    print('\nLoading model')
    device = 'cuda'
    model_config = ModelConfig(config['model_name'], use_pretrained=True)
    model = ModelFactory.create_feature_extractor(model_config)
    model.to(device)
    
    print('Loading data')
    transform = transforms.Compose([
        ModelFactory.get_resize_transform(config['model_name']),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.05)),
        transforms.ToTensor(),
        ModelFactory.get_normalization_transform(config['model_name']),
    ])
    train_loader = _get_dataloader(
        data_dir=config['data']['train_dir'],
        transform=transform,
        config=config,
        is_triplet=True,
        shuffle=True,
    )
    val_loader = _get_dataloader(
        data_dir=config['data']['test_dir'],
        transform=transform,
        config=config,
        is_triplet=True,
        is_deterministic=True,
    )
    val_eer_loader = _get_dataloader(
        data_dir=config['data']['test_dir'],
        transform=transform,
        config=config,
        is_triplet=False,
    )
    
    print('Training model')
    weights, epoch_metadata = _fit_model(model, device, train_loader, val_loader, val_eer_loader, config)
    
    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    model_dir = os.path.join(OUT_DIR, 'triplet-loss-open-world', config['experiment_name'], current_datetime)
    os.makedirs(os.path.join(model_dir, 'meta'), exist_ok=True)
    
    print('Saving model weights')
    save_file(weights, os.path.join(model_dir, f'{config['model_name']}.safetensors'))
    
    print('Saving metadata')
    best_epoch = min(epoch_metadata, key=lambda x: x['val_eer'])['epoch']
    train_images = list(set(train_loader.dataset.image_filenames.tolist()))
    val_images = list(set(val_loader.dataset.image_filenames.tolist()))
    metadata = {
        'config': config,
        'best_epoch': best_epoch,
        'epochs': epoch_metadata,
        'num_train_images': len(train_loader.dataset),
        'num_val_images': len(val_loader.dataset),
        'train_images': train_images,
        'val_images': val_images,
    }
    with open(os.path.join(model_dir, 'meta', 'train.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return model_dir

def _fit_model(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_eer_loader: DataLoader,
    config: dict
) -> tuple[dict, list[dict]]:
    scaler = GradScaler()
    optimizer = AdamP(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = TripletMarginLoss(margin=config['margin'], p=2)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['num_epochs'] // 10,
        T_mult=2,
        eta_min=config['learning_rate'] / 100,
    )
    
    epoch_metadata = []
    best_val_eer = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    num_epochs_since_best = 0
    
    for epoch in range(config['num_epochs']):
        total_train_loss = 0
        total_val_loss = 0
        
        model.train()
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} Training', leave=False)
        for anchor, positive, negative in train_pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device):
                anchor_features = model(anchor)
                positive_features = model(positive)
                negative_features = model(negative)
                
                loss = criterion(anchor_features, positive_features, negative_features)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss = loss.item()
            total_train_loss += loss
            train_pbar.set_postfix({'loss': f'{loss:.4f}'})
        
        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} Validation', leave=False)
            for anchor, positive, negative in val_pbar:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                with autocast(device):
                    anchor_features = model(anchor)
                    positive_features = model(positive)
                    negative_features = model(negative)
                
                    loss = criterion(anchor_features, positive_features, negative_features)
                
                loss = loss.item()
                total_val_loss += loss
                val_pbar.set_postfix({'loss': f'{loss:.4f}'})
                
            embeddings, labels = _extract_embeddings(model, device, val_eer_loader)
            fpr, tpr, _ = compute_distance_roc(embeddings, labels)
            val_eer = fpr[np.nanargmin(np.absolute((1 - tpr - fpr)))]
        
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Epoch, {epoch+1}/{config["num_epochs"]}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val EER: {val_eer:.6f}')
        
        epoch_metadata.append({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_eer': val_eer,
        })
        
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_weights = copy.deepcopy(model.state_dict())
            num_epochs_since_best = 0
        else:
            num_epochs_since_best += 1
            if num_epochs_since_best >= config['early_stopping_tolerance']:
                print(f'No improvement in {num_epochs_since_best} epochs, stopping training')
                break
        
    return best_weights, epoch_metadata
    
def eval_model(config, model_dir: str = None):
    print('\nLoading model')
    device = 'cuda'
    model_config = ModelConfig(config['model_name'])
    model = ModelFactory.create_feature_extractor(model_config)
    weights = load_file(os.path.join(model_dir, f'{config['model_name']}.safetensors'))
    model.load_state_dict(weights)
    model.to(device)
    
    transform = transforms.Compose([
        ModelFactory.get_resize_transform(config['model_name']),
        transforms.ToTensor(),
        ModelFactory.get_normalization_transform(config['model_name']),
    ])
    
    print('Evaluating model')
    print('Computing angle-vs-angle metrics')
    _eval_angles(model, device, transform, model_dir, False, config)
    print('Computing angle-vs-frontal metrics')
    _eval_angles(model, device, transform, model_dir, True, config)
    print('Computing t-SNE')
    _eval_tsne(model, device, transform, model_dir, config)
    
def _eval_angles(
    model: nn.Module,
    device: str,
    transform: transforms.Compose,
    model_dir: str,
    compare_with_frontal: bool,
    config: dict
):
    out_dir = os.path.join(model_dir, 'figs', 'angle-vs-frontal' if compare_with_frontal else 'angle-vs-angle')
    os.makedirs(out_dir, exist_ok=True)
    
    frontal_embeddings = None
    frontal_labels = None
    
    if compare_with_frontal:
        test_loader = _get_dataloader(
            data_dir=config['data']['test_dir'],
            transform=transform,
            config=config,
            is_triplet=False,
            angles=[FRONTAL_ANGLE],
        )
        frontal_embeddings, frontal_labels = _extract_embeddings(model, device, test_loader)
    
    results = {}
    pbar = tqdm(IRIS_IMAGE_ANGLES, desc='Evaluating', leave=False)
    for angle in pbar:
        pbar.set_postfix({'angle': angle})
        
        angled_embeddings = None
        angled_labels = None
        if compare_with_frontal and angle != FRONTAL_ANGLE or not compare_with_frontal:
            test_loader = _get_dataloader(
                data_dir=config['data']['test_dir'],
                transform=transform,
                config=config,
                is_triplet=False,
                angles=[angle],
            )
            angled_embeddings, angled_labels = _extract_embeddings(model, device, test_loader)
            
        angle_digits = str(int(angle.replace('N', '-').replace('P', '')))
        if compare_with_frontal and angle != FRONTAL_ANGLE:
            intra_distances, inter_distances = compute_inter_and_intra_class_distances(angled_embeddings, angled_labels, frontal_embeddings, frontal_labels)
            hist_filename = f'{angle.lower()}-vs-p00-dist-hist'
            
            fpr, tpr, thresholds = compute_distance_roc(angled_embeddings, angled_labels, frontal_embeddings, frontal_labels)
            key = f'{angle_digits} vs 0'
        else:
            if compare_with_frontal:
                angled_embeddings = frontal_embeddings
                angled_labels = frontal_labels
            
            intra_distances, inter_distances = compute_inter_and_intra_class_distances(angled_embeddings, angled_labels)
            hist_filename = f'{angle.lower()}-vs-p00-dist-hist' if compare_with_frontal else f'{angle.lower()}-dist-hist'
            
            fpr, tpr, thresholds = compute_distance_roc(angled_embeddings, angled_labels)
            key = f'{angle_digits} vs 0' if compare_with_frontal else angle_digits
            
        intra_inter_class_distance_histogram(
            intra_distances,
            inter_distances,
            out_dir,
            filename=hist_filename,
        )
            
        roc_auc = auc(fpr, tpr)
        eer = fpr[np.nanargmin(np.absolute((1 - tpr - fpr)))]
        results[key] = {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc,
            'eer': eer,
        }
    
    roc(
        results,
        out_dir,
        legend_title='Angles' if compare_with_frontal else 'Angle',
        filename='roc',
        include_auc=False,
        x_lim=(0.0, 0.02),
        y_lim=(0.98, 1.0),
    )
    
def _eval_tsne(
    model: nn.Module,
    device: str,
    transform: transforms.Compose,
    model_dir: str,
    config: dict
):
    test_loader = _get_dataloader(
        data_dir=config['data']['test_dir'],
        transform=transform,
        config=config,
        is_triplet=False,
    )
    embeddings, labels = _extract_embeddings(model, device, test_loader)
    
    out_dir = os.path.join(model_dir, 'figs')
    os.makedirs(out_dir, exist_ok=True)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    tsne_2d(
        embeddings_2d,
        labels,
        out_dir,
        filename='tsne-2d',
        legend_title='Subject',
    )
    
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings)
    tsne_3d(
        embeddings_3d,
        labels,
        out_dir,
        filename='tsne-3d',
        legend_title='Subject',
    )
        
def _extract_embeddings(model: nn.Module, device: str, loader: DataLoader) -> np.ndarray:
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            with autocast(device):
                embedding = model(x)
                
            embeddings.append(embedding.cpu())
            labels.append(y)
        
        embeddings = torch.cat(embeddings, dim=0).numpy()
        labels = torch.cat(labels, dim=0).numpy()
        return embeddings, labels
    
def _get_dataloader(
    data_dir: str,
    transform: transforms.Compose,
    config: dict,
    is_triplet: bool,
    is_deterministic: bool = False,
    shuffle: bool = False,
    angles: list[str] = None
) -> DataLoader:
    if is_triplet:
        dataset = IrisTripletDataset(
            data_dir=data_dir,
            transform=transform,
            angles=angles,
            angle_correction=config['angle_correction'],
            is_deterministic=is_deterministic,
            max_ancher_neg_angle_diff=config['max_anchor_negative_angle_diff'],
        )
    else:
        dataset = IrisDataset(
            data_dir=data_dir,
            transform=transform,
            angles=angles,
            angle_correction=config['angle_correction'],
        )
    
    num_workers = torch.get_num_threads()
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=torch.get_num_threads(),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else 0,
    )
    return loader