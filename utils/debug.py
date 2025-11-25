"""
Diagnostic utilities for TinyVLA
Focus on feature analysis for the cross-attention fusion model.
Includes tests for feature collapse, color, and positional/relative
feature representation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from typing import Dict, List
import logging

# Import model and dataset for main execution
try:
    from tiny_vla_model import create_tiny_vla
    from tiny_vla_dataset import BlockFindDataset
except ImportError:
    print("WARNING: Could not import tiny_vla_model or tiny_vla_dataset.")
    print("File will load, but the __main__ block will fail.")


logger = logging.getLogger(__name__)


# --- HELPER 1: Find block in image ---
def find_block_center(image_tensor: torch.Tensor, color_str: str) -> List[float]:
    """
    Finds the center (x, y) of the specified color block in an image tensor.
    Assumes image_tensor is (C, H, W) and normalized [0, 1].
    """
    # (H, W) = (64, 64)
    H, W = image_tensor.shape[1:]

    # Define target colors (assuming standard [1,0,0] for red, etc.)
    color_map_tensor = {
        'red': torch.tensor([1.0, 0.0, 0.0], device=image_tensor.device),
        'green': torch.tensor([0.0, 1.0, 0.0], device=image_tensor.device),
        'blue': torch.tensor([0.0, 0.0, 1.0], device=image_tensor.device),
        'yellow': torch.tensor([1.0, 1.0, 0.0], device=image_tensor.device),
    }

    if color_str not in color_map_tensor:
        return [-1.0, -1.0]

    target_rgb = color_map_tensor[color_str].view(3, 1, 1) # (3, 1, 1)

    # Calculate difference across all pixels
    # image_tensor (3, 64, 64) - target_rgb (3, 1, 1) -> (3, 64, 64)
    diff = torch.abs(image_tensor - target_rgb).sum(dim=0) # (64, 64)
    
    # Find pixels that match (with a small tolerance)
    mask = (diff < 0.1)
    
    # Get coordinates of matching pixels
    y_coords, x_coords = torch.where(mask)

    if len(y_coords) == 0:
        # Color not found in image
        return [-1.0, -1.0]

    # Return the center of the block
    center_y = y_coords.float().mean().item()
    center_x = x_coords.float().mean().item()
    
    return [center_x, center_y]

# --- HELPER 2: Map (x, y) to quadrant ---
def get_quadrant(position: List[float], grid_size: int = 64) -> str:
    """Maps an (x, y) coordinate to a quadrant category."""
    x, y = position
    mid_x = grid_size / 2.0
    mid_y = grid_size / 2.0
    
    if x < 0 or y < 0:
        return "unknown"

    if x < mid_x and y < mid_y:
        return "top-left"
    elif x >= mid_x and y < mid_y:
        return "top-right"
    elif x < mid_x and y >= mid_y:
        return "bottom-left"
    else: # x >= mid_x and y >= mid_y
        return "bottom-right"

# --- HELPER 3: Get relative direction ---
def get_relative_direction(pos_source: List[float], pos_target: List[float]) -> str:
    """
    Categorizes the direction from source to target as 'up', 'down', 'left', or 'right'.
    """
    if pos_source[0] < 0 or pos_target[0] < 0:
        return "unknown"
        
    x_source, y_source = pos_source
    x_target, y_target = pos_target
    
    dx = x_target - x_source
    dy = y_target - y_source
    
    if abs(dx) > abs(dy):
        # Move is primarily horizontal
        return "right" if dx > 0 else "left"
    else:
        # Move is primarily vertical
        # Note: In image coordinates, smaller y is 'up'
        return "down" if dy > 0 else "up"

# --- MAIN FEATURE EXTRACTION ---
def extract_fused_features(
    model,
    dataset,
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Extracts fused features and labels for:
    1. Target color
    2. Target quadrant
    3. Relative direction (Center to Target)
    """
    model.eval()

    features_list = []
    colors_list = []
    quadrant_list = []
    rel_dir_list = []
    images_list = []
    instructions_list = []

    # Check if the dataset is the expected type
    if not hasattr(dataset, 'samples'):
        print("ERROR: Dataset object does not have a '.samples' attribute.")
        print("This utility requires the BlockFindDataset, not a generic DataLoader or wrapper.")
        return {}

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            
            # Get tensors from __getitem__
            sample_tensors = dataset[i]
            image_tensor = sample_tensors['image'] # (C, H, W)
            
            # Get metadata from raw .samples list for robust labels
            try:
                sample_meta = dataset.samples[i]
                instruction_text = sample_meta['instruction']
            except (IndexError, AttributeError):
                print(f"Warning: Could not access dataset.samples[{i}]. Skipping.")
                continue

            # Tokenize
            if model.tokenizer is None:
                raise ValueError("Model tokenizer is not initialized.")
            
            tokens = model.tokenizer([instruction_text], return_tensors='pt', padding=True, truncation=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            # --- Perform full forward pass ---
            image_batch = image_tensor.unsqueeze(0).to(device)
            
            # Use shared encoder logic
            fused_final, _ = model.encode_vision_language(image_batch, input_ids, attention_mask)
            
            # Note: fused_final is already normalized and has residual added in encode_vision_language
            
            features_list.append(fused_final.cpu().numpy()[0])

            # Parse based on metadata, not instruction string ---
            try:
                # Get target color directly from the dataset's metadata
                color_target = sample_meta['target_color']
            except KeyError:
                print(f"Warning: Could not parse instruction: '{instruction_text}'")
                print("Dataset metadata missing 'target_color'. Attempting string parse.")
                # Fallback to string parsing if metadata fails
                try:
                    color_target = instruction_text.split()[2] # "find the [red] block"
                except IndexError:
                    print(f"Fallback parse failed for: '{instruction_text}'. Skipping.")
                    continue
            
            # Find the target block in the image
            pos_target = find_block_center(image_tensor, color_target)
            
            # Source is the center of the image
            H, W = image_tensor.shape[1:]
            pos_source = [W / 2.0, H / 2.0]
            
            # Get labels based on the target block
            quadrant = get_quadrant(pos_target, grid_size=H)
            rel_dir = get_relative_direction(pos_source, pos_target)
            
            colors_list.append(color_target) # Label is the target color
            quadrant_list.append(quadrant)   # Label is the target quadrant
            rel_dir_list.append(rel_dir)     # Label is the direction from center
            # --- END PARSING ---

            images_list.append(sample_tensors['image'].cpu().numpy()) # Use the tensor
            instructions_list.append(instruction_text)

    return {
        'features': np.array(features_list),
        'colors': np.array(colors_list),
        'quadrants': np.array(quadrant_list),
        'rel_dirs': np.array(rel_dir_list),
        'images': np.array(images_list),
        'instructions': np.array(instructions_list)
    }

# --- STATISTICAL & VISUALIZATION FUNCTIONS ---

def test_color_discrimination(features: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Statistical test: Are features different across label categories?
    (Generic ANOVA test for color, quadrant, or direction)
    
    Args:
        features: (N, feature_dim)
        labels: (N,) string labels (e.g., 'red', 'blue' or 'top-left')
    """
    unique_labels = np.unique(labels)
    
    # Ensure there's more than one group to compare
    if len(unique_labels) < 2:
        print(f"Warning: Only one label group found ('{unique_labels}'). Skipping ANOVA.")
        return {
            'f_statistics': np.array([]), 'p_values': np.array([]),
            'significant_dims': 0, 'total_dims': features.shape[1],
            'discrimination_rate': 0.0, 'color_means': {},
            'centroid_distances': {}
        }

    # Group features by label
    feature_groups = [features[labels == label] for label in unique_labels]

    # ANOVA for each feature dimension
    f_stats = []
    p_values = []

    for dim in range(features.shape[1]):
        dim_groups = [group[:, dim] for group in feature_groups]
        # Skip if any group is empty for this dimension (shouldn't happen)
        if any(len(g) == 0 for g in dim_groups):
            continue
            
        f_stat, p_val = stats.f_oneway(*dim_groups)
        f_stats.append(f_stat)
        p_values.append(p_val)

    if len(p_values) == 0:
        return {
            'f_statistics': np.array([]), 'p_values': np.array([]),
            'significant_dims': 0, 'total_dims': features.shape[1],
            'discrimination_rate': 0.0, 'color_means': {},
            'centroid_distances': {}
        }

    # Overall test: what fraction of dimensions discriminate labels?
    significant_dims = np.sum(np.array(p_values) < 0.05)
    total_dims = len(p_values)

    # Compute per-label mean features
    label_means = {label: features[labels == label].mean(axis=0)
                   for label in unique_labels}

    # Pairwise distances between label centroids
    centroid_distances = {}
    for i, c1 in enumerate(unique_labels):
        for c2 in unique_labels[i+1:]:
            dist = np.linalg.norm(label_means[c1] - label_means[c2])
            centroid_distances[f"{c1}_vs_{c2}"] = dist

    return {
        'f_statistics': np.array(f_stats),
        'p_values': np.array(p_values),
        'significant_dims': significant_dims,
        'total_dims': total_dims,
        'discrimination_rate': significant_dims / total_dims,
        'color_means': label_means,
        'centroid_distances': centroid_distances
    }


def visualize_feature_space(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'pca',
    save_path: str = 'feature_space.png',
    title_prefix: str = 'Features'
):
    """
    Visualize features in 2D using PCA or t-SNE
    Color points by label
    """
    # Reduce to 2D
    if method == 'pca':
        reducer = PCA(n_components=2)
        features_2d = reducer.fit_transform(features)
        explained_var = reducer.explained_variance_ratio_
        title = f'{title_prefix} (PCA)\nExplained variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}'
    elif method == 'tsne':
        # Perplexity must be less than n_samples
        perplexity = min(30, len(features) - 1)
        if perplexity <= 0:
            print(f"Warning: Not enough samples ({len(features)}) for t-SNE. Skipping.")
            return
            
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = reducer.fit_transform(features)
        title = f'{title_prefix} (t-SNE)'
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color map for blocks (add more as needed)
    color_map = {
        'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00', 'yellow': '#FFFF00',
        'top-left': '#FF00FF', 'top-right': '#00FFFF', 'bottom-left': '#FFA500', 'bottom-right': '#A52A2A',
        'up': '#FF69B4', 'down': '#8A2BE2', 'left': '#7FFF00', 'right': '#D2691E',
        'unknown': '#808080'
    }

    unique_labels = np.unique(labels)
    for label_name in unique_labels:
        mask = labels == label_name
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=color_map.get(label_name, f"#{np.random.randint(0, 0xFFFFFF):06x}"), # Random color if unknown
            label=label_name,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidths=1
        )

    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved feature space visualization to {save_path}")


def visualize_feature_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'feature_statistics.png'
):
    """
    Plot statistical properties of features by label
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    color_map = {
        'red': '#FF0000', 'blue': '#0000FF', 'green': '#00FF00', 'yellow': '#FFFF00',
    }
    unique_labels = np.unique(labels)

    # 1. Mean feature values per label
    ax = axes[0, 0]
    for label_name in unique_labels:
        mask = labels == label_name
        mean_features = features[mask].mean(axis=0)
        ax.plot(mean_features, label=label_name, color=color_map.get(label_name, 'gray'), linewidth=2)
    ax.set_xlabel('Feature Dimension', fontsize=10)
    ax.set_ylabel('Mean Feature Value', fontsize=10)
    ax.set_title('Mean Features by Label', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Feature variance per label
    ax = axes[0, 1]
    for label_name in unique_labels:
        mask = labels == label_name
        var_features = features[mask].var(axis=0)
        ax.plot(var_features, label=label_name, color=color_map.get(label_name, 'gray'), linewidth=2)
    ax.set_xlabel('Feature Dimension', fontsize=10)
    ax.set_ylabel('Feature Variance', fontsize=10)
    ax.set_title('Feature Variance by Label', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Feature distribution (histogram)
    ax = axes[1, 0]
    for label_name in unique_labels:
        mask = labels == label_name
        flat_features = features[mask].flatten()
        ax.hist(flat_features, bins=50, alpha=0.5, label=label_name,
                color=color_map.get(label_name, 'gray'))
    ax.set_xlabel('Feature Value', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Feature Value Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Pairwise label distances (heatmap)
    ax = axes[1, 1]
    n_labels = len(unique_labels)
    distance_matrix = np.zeros((n_labels, n_labels))

    label_means = {label: features[labels == label].mean(axis=0)
                   for label in unique_labels}

    for i, c1 in enumerate(unique_labels):
        for j, c2 in enumerate(unique_labels):
            distance_matrix[i, j] = np.linalg.norm(label_means[c1] - label_means[c2])

    im = ax.imshow(distance_matrix, cmap='viridis')
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(unique_labels, rotation=45, ha="right")
    ax.set_yticklabels(unique_labels)
    ax.set_title('Pairwise Centroid Distances', fontsize=12)

    # Add text annotations
    for i in range(n_labels):
        for j in range(n_labels):
            text = ax.text(j, i, f'{distance_matrix[i, j]:.4f}',
                          ha="center", va="center", color="white", fontsize=10)

    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved feature statistics to {save_path}")


def visualize_patch_features(
    model,
    dataset,
    sample_idx: int = 0,
    save_path: str = 'patch_features.png',
    device: str = 'cuda'
):
    """
    Visualize what the vision encoder sees at the patch level
    Shows original image + heatmap of patch feature norms
    """
    model.eval()

    # Get instruction from metadata for title
    try:
        sample_meta = dataset.samples[sample_idx]
        instruction_text = sample_meta['instruction']
    except Exception:
        instruction_text = "N/A" # Fallback
        
    sample = dataset[sample_idx]
    image = sample['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        # Get patch embeddings (before transformer)
        vision_enc = model.vision_encoder

        # Conv projection to patches
        x = vision_enc.patch_embed(image)  # (B, embed_dim, H/P, W/P)

        # Flatten and transpose
        x_flat = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Get feature norms per patch
        patch_norms = torch.norm(x_flat, dim=2).cpu().numpy()[0]  # (num_patches,)

        # Reshape to grid
        num_patches_per_side = int(np.sqrt(len(patch_norms)))
        patch_grid = patch_norms.reshape(num_patches_per_side, num_patches_per_side)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Convert (C, H, W) tensor to (H, W, C) numpy array
    img_to_plot = sample['image'].permute(1, 2, 0).cpu().numpy()

    # Original image
    axes[0].imshow(img_to_plot) # Use the permuted image
    axes[0].set_title(f"Original Image\n{instruction_text[:30]}...", fontsize=10)
    axes[0].axis('off')

    # Patch feature norms
    im = axes[1].imshow(patch_grid, cmap='hot', interpolation='nearest')
    axes[1].set_title('Patch Feature Magnitudes\n(after embedding)', fontsize=10)
    plt.colorbar(im, ax=axes[1])

    # Overlay on image
    axes[2].imshow(img_to_plot) # Use the permuted image
    im2 = axes[2].imshow(patch_grid, cmap='hot', alpha=0.5, interpolation='bilinear',
                         extent=[0, 64, 64, 0])
    axes[2].set_title('Overlay: Feature Magnitude\non Original Image', fontsize=10)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved patch feature visualization to {save_path}")


# --- MAIN DIAGNOSTIC SUITES ---

def run_full_diagnostics(
    model,
    dataset,
    num_samples: int = 200,
    device: str = 'cuda',
    output_prefix: str = 'diagnostics_fused'
):
    """
    Run complete diagnostic suite on *fused* features
    Now includes color, position, and relative direction analysis.
    """
    print("="*60)
    print("Running Full Model (Fusion) Diagnostics")
    print("="*60)

    # Extract features
    print(f"\nExtracting *fused* features from {num_samples} samples...")
    data = extract_fused_features(model, dataset, num_samples, device)
    if not data: # Handle empty return on error
        print("Feature extraction failed. Aborting diagnostics.")
        return {}
        
    features = data['features']
    colors = data['colors']
    quadrants = data['quadrants']
    rel_dirs = data['rel_dirs']

    if len(features) == 0:
        print("No features were extracted. Aborting diagnostics.")
        return {}

    print(f"Fused feature shape: {features.shape}")
    print(f"Unique colors: {np.unique(colors)}")
    print(f"Unique quadrants: {np.unique(quadrants)}")
    print(f"Unique relative directions: {np.unique(rel_dirs)}")

    # --- Test 1: Color Discrimination ---
    print("\n" + "="*60)
    #  Clarified test name
    print("Test 1: Target Color Discrimination (ANOVA)")
    print("="*60)
    stats_results_color = test_color_discrimination(features, colors)
    print(f"Discrimination rate: {stats_results_color['discrimination_rate']:.2%}")

    # --- Test 2: Positional Discrimination ---
    print("\n" + "="*60)
    #  Clarified test name
    print("Test 2: Target Position Discrimination (ANOVA)")
    print("="*60)
    stats_results_pos = test_color_discrimination(features, quadrants)
    print(f"Discrimination rate: {stats_results_pos['discrimination_rate']:.2%}")

    # --- Test 3: Relative Direction Discrimination ---
    print("\n" + "="*60)
    #  Clarified test name
    print("Test 3: Relative Direction (from Center) Discrimination (ANOVA)")
    print("="*60)
    stats_results_dir = test_color_discrimination(features, rel_dirs)
    print(f"Discrimination rate: {stats_results_dir['discrimination_rate']:.2%}")
    print("\nPairwise centroid distances (by direction):")
    for pair, dist in stats_results_dir['centroid_distances'].items():
        print(f"  {pair}: {dist:.4f}")

    print("\n" + "="*60)
    print("Overall Feature Statistics")
    print("="*60)
    feature_mean = features.mean()
    feature_std = features.std()
    print(f"  Mean: {feature_mean:.6f}")
    print(f"  Std:  {feature_std:.6f}")

    # Visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    visualize_feature_space(features, colors, method='pca',
                           save_path=f'{output_prefix}_pca_by_color.png',
                           title_prefix='Fused Features by Target Color')
    visualize_feature_space(features, quadrants, method='pca',
                           save_path=f'{output_prefix}_pca_by_position.png',
                           title_prefix='Fused Features by Target Position')
    visualize_feature_space(features, rel_dirs, method='pca',
                           save_path=f'{output_prefix}_pca_by_direction.png',
                           title_prefix='Fused Features by Relative Direction')
    
    # Generate stats plot only for color (most interpretable)
    visualize_feature_statistics(features, colors,
                                save_path=f'{output_prefix}_stats_by_color.png')

    visualize_patch_features(model, dataset, sample_idx=0,
                            save_path=f'{output_prefix}_patches_0.png',
                            device=device)
    visualize_patch_features(model, dataset, sample_idx=1,
                            save_path=f'{output_prefix}_patches_1.png',
                            device=device)

    print("\n" + "="*60)
    print("Diagnostics Complete!")
    print("="*60)

    # Summary judgment
    print("\nSummary:")
    #  Changed "source" to "target"
    if stats_results_color['discrimination_rate'] > 0.5:
        print("✓ Fused features CAN discriminate *target color*")
    else:
        print("✗ Fused features CANNOT discriminate *target color*")
        
    if stats_results_pos['discrimination_rate'] > 0.5:
        print("✓ Fused features CAN discriminate *target position*")
    else:
        print("✗ Fused features CANNOT discriminate *target position*")

    if stats_results_dir['discrimination_rate'] > 0.5:
        print("✓ Fused features CAN discriminate *relative direction*")
    elif stats_results_dir['discrimination_rate'] > 0.1:
        print("⚠ Fused features have WEAK *relative direction* discrimination")
    else:
        print("✗ Fused features CANNOT discriminate *relative direction*")

    if feature_std < 0.1:
        print("✗ Features appear COLLAPSED (std < 0.1)")
    else:
        print("✓ Features have reasonable variance")

    return {
        'features': features,
        'colors': colors,
        'quadrants': quadrants,
        'rel_dirs': rel_dirs,
        'stats_color': stats_results_color,
        'stats_pos': stats_results_pos,
        'stats_dir': stats_results_dir
    }


def compare_vision_language_magnitudes(
    model,
    dataset,
    num_samples: int = 100,
    device: str = 'cuda'
) -> Dict:
    """
    Compare magnitudes (Mean and Std) of activation values
    at each stage of the fusion process.
    Checks for "noise amplification" from LayerNorm.
    """
    model.eval()

    lang_query_means, lang_query_stds = [], []
    vision_patch_means, vision_patch_stds = [], []
    
    # We will measure BOTH pre-norm and post-norm
    fused_pre_norm_means, fused_pre_norm_stds = [], []
    fused_post_norm_means, fused_post_norm_stds = [], []


    # Check if the dataset is the expected type
    if not hasattr(dataset, 'samples'):
        print("ERROR: Dataset object does not have a '.samples' attribute.")
        return {}

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            #  Get instruction from metadata
            try:
                instruction = [dataset.samples[i]['instruction']]
            except Exception:
                continue # Skip if error
            
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)

            # Tokenize
            if model.tokenizer is None:
                raise ValueError("Model tokenizer is not initialized.")
            
            tokens = model.tokenizer(instruction, return_tensors='pt', padding=True, truncation=True)
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            
            # --- Get features at each stage ---
            
            # 1. Language Query
            lang_features = model.language_model(input_ids, attention_mask)
            lang_pooled = lang_features[:, 0, :]      # (1, lang_dim)

            # 2. Vision Patches (Key/Value)
            vision_features = model.vision_encoder(image)
            vision_patches = vision_features[:, 1:, :]
            vision_patches_proj = model.vision_proj(vision_patches)
            vision_patches_norm = model.vision_norm(vision_patches_proj) # (1, num_patches, lang_dim)

            # 3. Fused Output (PRE-NORM)
            fused_features, _ = model.fusion_attention(
                query=lang_pooled.unsqueeze(1),
                key=vision_patches_norm,
                value=vision_patches_norm
            )
            fused_pre_norm = fused_features.squeeze(1) # (1, lang_dim)
            
            # 4. Fused Output (POST-NORM)
            # This assumes your model has the residual connection and final norm
            try:
                fused_with_residual = lang_pooled + fused_pre_norm 
                fused_post_norm = model.fusion_output_norm(fused_with_residual)
            except AttributeError:
                print("ERROR: Model does not seem to have fusion_output_norm.")
                print("Skipping compare_vision_language_magnitudes.")
                return {} # Return empty
            
            # --- Compute mean and std of the *values* ---
            lang_query_means.append(lang_pooled.mean().item())
            lang_query_stds.append(lang_pooled.std().item())
            
            vision_patch_means.append(vision_patches_norm.mean().item())
            vision_patch_stds.append(vision_patches_norm.std().item())
            
            # Add stats for BOTH pre and post
            fused_pre_norm_means.append(fused_pre_norm.mean().item())
            fused_pre_norm_stds.append(fused_pre_norm.std().item())
            
            fused_post_norm_means.append(fused_post_norm.mean().item())
            fused_post_norm_stds.append(fused_post_norm.std().item())


    # Convert lists to numpy arrays for aggregate stats
    lang_query_stds = np.array(lang_query_stds)
    vision_patch_stds = np.array(vision_patch_stds)
    fused_pre_norm_stds = np.array(fused_pre_norm_stds)
    fused_post_norm_stds = np.array(fused_post_norm_stds)

    # Compute statistics
    results = {
        'lang_query_std_val': lang_query_stds.mean(),
        'vision_patch_std_val': vision_patch_stds.mean(),
        'fused_pre_norm_std_val': fused_pre_norm_stds.mean(),
        'fused_post_norm_std_val': fused_post_norm_stds.mean()
    }

    # Print summary
    print("="*60)
    print("Activation Statistics (Mean/Std of Values)")
    print("============================================================")
    print("\nLanguage Query (to attention):")
    print(f"  Std (value):  {results['lang_query_std_val']:.4f}")
    print("\nVision Patches (to attention):")
    print(f"  Std (value):  {results['vision_patch_std_val']:.4f}")
    
    print("\nFused features (PRE-NORM, from attention):")
    print(f"  Std (value):  {results['fused_pre_norm_std_val']:.4f}")

    print("\nFused features (POST-RESIDUAL, PRE-NORM):")
    print("  Note: Std of (lang_pooled + fused_pre_norm) is now passed to norm layer.")

    print("\nFused features (POST-NORM, to action head):")
    print(f"  Std (value):  {results['fused_post_norm_std_val']:.4f}")


    if results['fused_pre_norm_std_val'] < 0.1:
        print("\n⚠ WARNING: Attention output (pre-residual) is near-zero (std < 0.1).")
        print("   The model might be ignoring the fusion and only using the residual.")
    else:
        print("\n✓ Attention output (pre-residual) has a strong signal.")

    return results


def check_projection_layer_impact(
    model,
    dataset,
    num_samples: int = 200,
    device: str = 'cuda'
) -> Dict:
    """
    Check if vision projection layer destroys color discrimination.
    Compares *mean of patch features* before and after vision_proj.
    """
    model.eval()

    features_before_proj = []
    features_after_proj = []
    colors = []

    # Check if the dataset is the expected type
    if not hasattr(dataset, 'samples'):
        print("ERROR: Dataset object does not have a '.samples' attribute.")
        return {}

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample_tensors = dataset[i]
            image_tensor = sample_tensors['image'] # (C, H, W)
            
            #  Get metadata from raw .samples list for robust labels
            try:
                sample_meta = dataset.samples[i]
            except (IndexError, AttributeError):
                continue

            # Get vision features BEFORE projection
            vision_raw = model.vision_encoder(image_tensor.unsqueeze(0).to(device))
            
            # Get patch features (skip CLS)
            vision_patches_before = vision_raw[:, 1:, :]  # (1, num_patches, vision_dim)
            
            # Get MEAN of patches BEFORE
            vision_mean_before = vision_patches_before.mean(dim=1) # (1, vision_dim)

            # Project ALL patches
            vision_patches_projected = model.vision_proj(vision_patches_before)
            vision_patches_after = model.vision_norm(vision_patches_projected)
            
            # Get MEAN of patches AFTER
            vision_mean_after = vision_patches_after.mean(dim=1) # (1, lang_dim)

            features_before_proj.append(vision_mean_before.cpu().numpy()[0])
            features_after_proj.append(vision_mean_after.cpu().numpy()[0])

            # Get target color from metadata
            try:
                color = sample_meta['target_color']
            except KeyError:
                continue # Skip malformed sample
            colors.append(color)

    features_before_proj = np.array(features_before_proj)
    features_after_proj = np.array(features_after_proj)
    colors = np.array(colors)

    # Test color discrimination before and after projection
    print("="*60)
    print("Checking Vision Projection Layer Impact (on Mean-Pooled Patches)")
    print("="*60)

    print(f"\n[BEFORE projection] Vision encoder *mean patch* output ({features_before_proj.shape[1]} dims):")
    stats_before = test_color_discrimination(features_before_proj, colors)
    print(f"  Discrimination rate: {stats_before['discrimination_rate']:.2%}")

    print(f"\n[AFTER projection] Vision projection *mean patch* output ({features_after_proj.shape[1]} dims):")
    stats_after = test_color_discrimination(features_after_proj, colors)
    print(f"  Discrimination rate: {stats_after['discrimination_rate']:.2%}")

    # Compute average distance change
    before_dists = list(stats_before['centroid_distances'].values())
    after_dists = list(stats_after['centroid_distances'].values())
    
    avg_before = np.mean(before_dists) if before_dists else 0.0
    avg_after = np.mean(after_dists) if after_dists else 0.0
    ratio = (avg_after / avg_before) if avg_before > 1e-6 else 1.0
    
    print("\n[SUMMARY]")
    print(f"  Average centroid distance BEFORE: {avg_before:.4f}")
    print(f"  Average centroid distance AFTER:  {avg_after:.4f}")
    print(f"  Ratio (after/before): {ratio:.2f}x")

    if avg_before < 1e-6:
         print("\n⚠ WARNING: No color separation detected BEFORE projection.")
    elif avg_after < avg_before * 0.5:
        print("\n✗ Projection layer is DESTROYING color information!")
        print(f"   Distances decreased by {(1 - ratio)*100:.1f}%")
    elif avg_after < avg_before * 0.9:
        print("\n⚠ Projection layer is WEAKENING color information")
        print(f"   Distances decreased by {(1 - ratio)*100:.1f}%")
    else:
        print("\n✓ Projection layer preserves/enhances color information")

    return {
        'features_before': features_before_proj,
        'features_after': features_after_proj,
        'stats_after': stats_after,
        'colors': colors
    }


def check_text_generation(
    model,
    dataset,
    num_samples: int = 5,
    device: str = 'cuda'
) -> None:
    """
    Check text generation capabilities of the model.
    Prints sample inputs, ground truth, and generated text.
    """
    print("="*60)
    print("Checking Text Generation")
    print("="*60)
    
    model.eval()
    
    # Check if model has text decoder
    if not getattr(model, 'use_text_decoder', False) or model.text_decoder is None:
        print("Model does not have a text decoder enabled. Skipping.")
        return

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        instruction = sample['instruction']
        description_gt = sample.get('description', 'N/A')
        action_gt = sample['action'].numpy()
        
        # Prepare inputs
        _, input_ids, attention_mask = model.prepare_inputs(image, [instruction])
        
        # Generate text
        with torch.no_grad():
            generated_ids = model.generate_text(image, input_ids, attention_mask)
            generated_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        print(f"\nSample {i+1}:")
        print(f"  Instruction:  {instruction}")
        print(f"  Action GT:    [{action_gt[0]:.2f}, {action_gt[1]:.2f}]")
        print(f"  Text GT:      {description_gt}")
        print(f"  Generated:    {generated_text}")
        
    print("\n" + "="*60)


if __name__ == "__main__":
    # Example usage
    print("Loading model and dataset...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load trained model
    model = None
    dataset = None
    
    try:
        import json
        config_path = 'checkpoints/config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        model = create_tiny_vla(config.get('model'))
    except Exception as e:
        print(f"ERROR: Could not create model: {e}")
        exit()
        
    try:
        checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("WARNING: 'checkpoints/best_model.pt' not found. Running with initialized weights.")
        model = model.to(device)
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        model = model.to(device) # Continue with initialized model

    # Create test dataset
    try:
        dataset = BlockFindDataset(num_samples=500, seed=44)
        print("Loaded BlockFindDataset.")
    except Exception as e:
        print(f"ERROR: Could not load BlockFindDataset: {e}")
        print("Please ensure 'tiny_vla_dataset.py' is in the same directory.")
        exit()

    if model and dataset:
        # Check projection layer impact
        print("\n" + "="*60)
        print("="*60)
        proj_results = check_projection_layer_impact(model, dataset, num_samples=200, device=device)

        # Run full diagnostics
        print("\n" + "="*60)
        print("="*60)
        results = run_full_diagnostics(model, dataset, num_samples=200, device=device)

        # Compare vision vs language magnitudes
        print("\n" + "="*60)
        print("="*60)
        magnitude_results = compare_vision_language_magnitudes(model, dataset, num_samples=200, device=device)

        # Check text generation
        print("\n" + "="*60)
        print("="*60)
        check_text_generation(model, dataset, num_samples=5, device=device)
