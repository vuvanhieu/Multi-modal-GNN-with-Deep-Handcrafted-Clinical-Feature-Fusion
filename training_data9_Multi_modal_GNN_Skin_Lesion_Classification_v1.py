import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from sklearn.cluster import KMeans
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

def prepare_clinical_encoder(metadata_df):
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # ✅ Chuẩn hóa cột phù hợp ISIC 2020
    if 'image_name' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image_name': 'image_id'})
    elif 'image' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image': 'image_id'})

    # ✅ Đặt lại cột anatom_site chung
    if 'anatom_site_general_challenge' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general_challenge']
    elif 'anatom_site_general' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general']
    else:
        metadata_df['anatom_site'] = "unknown"

    # ✅ Điền missing values
    metadata_df['sex'] = metadata_df['sex'].fillna("unknown")
    metadata_df['anatom_site'] = metadata_df['anatom_site'].fillna("unknown")
    metadata_df['age_approx'] = metadata_df['age_approx'].fillna(metadata_df['age_approx'].mean())

    # ✅ Tạo encoder và scaler
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    one_hot_encoder.fit(metadata_df[['sex', 'anatom_site']])

    age_scaler = StandardScaler()
    age_scaler.fit(metadata_df[['age_approx']])

    return one_hot_encoder, age_scaler



def extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler):
    import re

    # ✅ Trích xuất image_id từ tên file
    image_ids = [re.search(r'(ISIC_\d+)', fname).group(1) if re.search(r'(ISIC_\d+)', fname) else None for fname in file_names]
    image_ids_clean = [img for img in image_ids if img is not None]

    # ✅ Chuẩn hóa metadata
    if 'image_name' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image_name': 'image_id'})
    elif 'image' in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={'image': 'image_id'})

    if 'anatom_site_general_challenge' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general_challenge']
    elif 'anatom_site_general' in metadata_df.columns:
        metadata_df['anatom_site'] = metadata_df['anatom_site_general']
    else:
        metadata_df['anatom_site'] = "unknown"

    metadata_df['sex'] = metadata_df['sex'].fillna("unknown")
    metadata_df['anatom_site'] = metadata_df['anatom_site'].fillna("unknown")
    metadata_df['age_approx'] = metadata_df['age_approx'].fillna(age_scaler.mean_[0])

    matched_df = metadata_df[metadata_df['image_id'].isin(image_ids_clean)].copy()
    if matched_df.empty:
        print("⚠️ Warning: No matching clinical metadata found.")
        return np.zeros((len(file_names), one_hot_encoder.transform([['unknown', 'unknown']]).shape[1] + 1))

    matched_df = matched_df.set_index('image_id').reindex(image_ids_clean)

    cat_feats = one_hot_encoder.transform(matched_df[['sex', 'anatom_site']].fillna("unknown"))
    age_feat = age_scaler.transform(matched_df[['age_approx']])

    return np.hstack([cat_feats, age_feat])


def extract_clinical_features(metadata_df, file_names, one_hot_encoder, age_scaler):
    import re

    # Lấy image_id từ tên file
    image_ids = [re.match(r"(ISIC_\d+)", fn).group(1) for fn in file_names if re.match(r"(ISIC_\d+)", fn)]

    # Chọn các hàng tương ứng
    metadata_df = metadata_df.set_index("image_id")
    matched_df = metadata_df.loc[image_ids].copy()

    # Xử lý thiếu (phòng khi có ảnh chưa có metadata)
    matched_df['sex'] = matched_df['sex'].fillna('unknown')
    matched_df['localization'] = matched_df['localization'].fillna('unknown')
    matched_df['age'] = matched_df['age'].fillna(age_scaler.mean_[0])

    # One-hot encode
    cat_features = one_hot_encoder.transform(matched_df[['sex', 'localization']])
    age_feature = age_scaler.transform(matched_df[['age']])

    clinical_features = np.hstack([cat_features, age_feature])
    return clinical_features



def plot_label_distribution(y_encoded, label_encoder, save_path, title="Label Distribution After Augmentation"):
    # Tạo Series tên nhãn tương ứng
    label_names = dict(enumerate(label_encoder.classes_))
    label_series = pd.Series([label_names[label] for label in y_encoded])

    # Lên bảng đếm và sắp xếp
    value_counts = label_series.value_counts().sort_index()
    sorted_labels = sorted(value_counts.index.tolist())  # Giữ thứ tự ABC

    # Ánh xạ màu từ colormap
    palette = sns.color_palette("tab10", len(sorted_labels))

    # Vẽ biểu đồ
    plt.figure(figsize=(8, 5))
    sns.countplot(x=label_series, order=sorted_labels, palette=palette)
    # plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"📊 Multi-class label distribution plot saved to: {save_path}")
    
def augment_all_classes_to_balance(X, y, noise_std=0.01):
    """
    Tự động tăng mẫu cho tất cả các lớp để cân bằng với lớp có số lượng nhiều nhất.
    Áp dụng Gaussian noise để tạo mẫu synthetic.

    Args:
        X: ndarray (n_samples, n_features)
        y: ndarray (n_samples,)
        noise_std: độ lệch chuẩn của Gaussian noise

    Returns:
        X_balanced, y_balanced: sau tăng cường
    """
    from collections import Counter
    import numpy as np

    class_counts = Counter(y)
    max_count = max(class_counts.values())
    print(f"📊 Class distribution before augmentation: {dict(class_counts)}")

    X_list, y_list = [X], [y]

    for cls in class_counts:
        count = class_counts[cls]
        needed = max_count - count
        if needed > 0:
            X_cls = X[y == cls]
            synthetic = []
            for _ in range(needed):
                idx = np.random.randint(0, len(X_cls))
                noisy = X_cls[idx] + np.random.normal(0, noise_std, size=X.shape[1])
                synthetic.append(noisy)
            X_list.append(np.array(synthetic))
            y_list.append(np.full(needed, cls))
            print(f"✅ Augmented class {cls} with {needed} synthetic samples")

    X_bal = np.vstack(X_list)
    y_bal = np.hstack(y_list)
    return X_bal, y_bal



def normalize_data(train_data, test_data):
    """
    Normalize the data using StandardScaler after replacing NaN values.
    """
    scaler = StandardScaler()

    # Nếu dữ liệu chứa NaN, thay thế bằng giá trị trung bình của từng cột
    if np.isnan(train_data).sum() > 0:
        print(f"⚠️ Warning: Found NaN in train_data. Replacing with column means.")
        col_mean = np.nanmean(train_data, axis=0)
        train_data = np.where(np.isnan(train_data), col_mean, train_data)

    if np.isnan(test_data).sum() > 0:
        print(f"⚠️ Warning: Found NaN in test_data. Replacing with column means.")
        col_mean = np.nanmean(test_data, axis=0)
        test_data = np.where(np.isnan(test_data), col_mean, test_data)

    train_data_normalized = scaler.fit_transform(train_data)
    test_data_normalized = scaler.transform(test_data)

    return train_data_normalized, test_data_normalized


def create_graph(features, labels, train_idx=None, test_idx=None, k=5, use_mask=True):
    """
    Tạo full graph có kết nối toàn bộ KNN (bỏ clustering) và gán train/test mask nếu được chỉ định.
    Nếu use_mask=False: hoạt động như trước, sử dụng clustering subgraph.
    Nếu use_mask=True: tạo 1 đồ thị lớn và gán train/test mask.
    """
    if use_mask and train_idx is not None and test_idx is not None:
        # ✅ FULL GRAPH WITH MASK
        knn_graph = kneighbors_graph(features, n_neighbors=k, mode='connectivity', include_self=False)
        edges = [(i, j) for i, j in zip(*knn_graph.nonzero())]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        train_mask = torch.zeros(len(labels), dtype=torch.bool)
        test_mask = torch.zeros(len(labels), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

    else:
        # 🔄 Subgraph-based (old logic)
        features_transformed = features
        num_clusters = max(2, min(len(features) // 10, 10))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_transformed)
        edges = []

        for cluster in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) > 1:
                subgraph_features = features_transformed[cluster_indices]
                effective_k = min(k, len(cluster_indices) - 1)
                if effective_k > 0:
                    knn_graph = kneighbors_graph(subgraph_features, n_neighbors=effective_k, mode='connectivity')
                    for i, j in zip(*knn_graph.nonzero()):
                        edges.append((cluster_indices[i], cluster_indices[j]))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)



import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.utils.class_weight import compute_class_weight

def train_gnn_model(model, data, optimizer, epochs, epoch_result_out, patience=10, num_atoms=0, alpha=1.0, use_sparse_coding=False, device="cpu"):
    model.to(device)
    data = data.to(device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # ✅ Tính class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(data.y.cpu().numpy()),
        y=data.y.cpu().numpy()
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        train_loss = F.nll_loss(out, data.y, weight=weight_tensor)  # ✅ áp dụng weight
        train_loss.backward()
        optimizer.step()

        train_preds = out.argmax(dim=1)
        train_acc = (train_preds == data.y).sum().item() / data.y.size(0)

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = F.nll_loss(val_out, data.y, weight=weight_tensor).item()  # ✅ val loss có weight
            val_preds = val_out.argmax(dim=1)
            val_acc = (val_preds == data.y).sum().item() / data.y.size(0)

        history["train_loss"].append(train_loss.item())
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            best_model_path = os.path.join(epoch_result_out, "best_gnn_model.pth")
            torch.save(best_model_state, best_model_path)
            print(f"💾 Best model saved at epoch {epoch + 1} with val loss {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict(torch.load(best_model_path))
        print("✅ Best model loaded with val loss:", best_loss)

    return model, history


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

def plot_combined_metrics(metric_collection, result_folder):
    """
    Plot combined metrics such as Accuracy, Precision, Recall, F1-Score, Sensitivity, Specificity,
    and Training Time for all models across different batch sizes.

    Each batch size gets its own chart.
    
    Parameters:
    - metric_collection: List of dictionaries containing model evaluation results.
    - result_folder: Folder to save the metric comparison plots.
    """
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(metric_collection)

    # Define the relevant metrics for comparison
    possible_metrics = [
        "Test Accuracy", "Precision", "Recall", "F1 Score", 
        "Sensitivity", "Specificity", "Training Time (s)"
    ]

    # Get available metrics in the dataset
    available_metrics = [metric for metric in possible_metrics if metric in df.columns]
    if not available_metrics:
        print("⚠️ No valid metrics found for plotting. Skipping combined metric plots.")
        return

    # Ensure output directory exists
    os.makedirs(result_folder, exist_ok=True)

    # Get unique batch sizes
    batch_sizes = df["Batch Size"].unique()

    for batch_size in batch_sizes:
        df_batch = df[df["Batch Size"] == batch_size]  # Filter data by batch size

        batch_folder = os.path.join(result_folder, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)

        for metric in available_metrics:
            plt.figure(figsize=(10, 6))

            # Aggregate data for plotting
            grouped_data = df_batch.groupby(["Model"])[metric].mean().reset_index()
            models = grouped_data["Model"].unique()

            # Ensure metric values are numeric
            grouped_data[metric] = pd.to_numeric(grouped_data[metric], errors='coerce')

            # Sort models based on metric value for better visualization
            grouped_data = grouped_data.sort_values(by=metric, ascending=False)

            # Plot bars
            plt.barh(grouped_data["Model"], grouped_data[metric], color="blue", alpha=0.7)

            # Add value labels to bars
            for index, value in enumerate(grouped_data[metric]):
                plt.text(value, index, f"{value:.4f}", va="center", fontsize=10, color="black")

            plt.xlabel(metric)
            plt.ylabel("Model")
            plt.title(f"{metric} Comparison (Batch Size: {batch_size})")
            plt.grid(axis="x", linestyle="--", alpha=0.5)

            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(batch_folder, f"{metric.lower().replace(' ', '_')}_batch_size_{batch_size}_comparison.png"))
            plt.close()

    print("✅ All combined metric comparison plots saved successfully!")


def plot_epoch_based_metrics(all_histories, result_dir):
    """
    Vẽ biểu đồ timeline của Train Loss, Validation Loss, Train Accuracy, Validation Accuracy
    theo các giá trị batch_size, hỗ trợ cả MLP và GNN.
    """
    metrics_list = []

    for model_name, model_histories in all_histories.items():
        for history_entry in model_histories:
            batch_size = history_entry.get("batch_size", 32)
            epoch = history_entry.get("epoch", 100)
            history = history_entry.get("history", {})

            # Xác định key loss/accuracy phù hợp (GNN vs MLP)
            if "train_loss" in history:
                train_loss_key = "train_loss"
                train_acc_key = "train_accuracy"
            else:
                train_loss_key = "loss"
                train_acc_key = "accuracy"

            val_loss_key = "val_loss"
            val_acc_key = "val_accuracy"

            # Kiểm tra đầy đủ các keys
            required_keys = [train_loss_key, val_loss_key, train_acc_key, val_acc_key]
            if not all(k in history for k in required_keys):
                print(f"⚠️ Skipping history for model {model_name} due to missing keys.")
                continue

            for epoch_idx, (train_loss, val_loss, train_acc, val_acc) in enumerate(
                zip(history[train_loss_key], history[val_loss_key],
                    history[train_acc_key], history[val_acc_key])
            ):
                metrics_list.append({
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch_idx + 1,
                    "Train Loss": train_loss,
                    "Validation Loss": val_loss,
                    "Train Accuracy": train_acc,
                    "Validation Accuracy": val_acc,
                })

    if not metrics_list:
        print("⚠️ No valid training histories found. Skipping timeline plotting.")
        return

    df = pd.DataFrame(metrics_list)
    metrics = ["Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]

    batch_sizes = df["Batch Size"].unique()
    for batch_size in batch_sizes:
        batch_folder = os.path.join(result_dir, f"batch_size_{batch_size}")
        os.makedirs(batch_folder, exist_ok=True)

        for metric in metrics:
            plt.figure(figsize=(14, 8))
            batch_df = df[df["Batch Size"] == batch_size]
            for model_name, model_df in batch_df.groupby("Model"):
                grouped = model_df.groupby("Epoch")[metric].mean().reset_index()
                epochs = grouped["Epoch"]
                metric_values = grouped[metric]
                plt.plot(epochs, metric_values, label=model_name, marker='o', linestyle='-')

            plt.xlabel("Epochs", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend(title="Models", loc="best", fontsize=10)
            plt.tight_layout()

            plot_path = os.path.join(batch_folder, f"{metric.lower().replace(' ', '_')}_batch_size_{batch_size}_timeline_comparison.png")
            plt.savefig(plot_path)
            plt.close()

    print(f"📊 Epoch-based timeline comparison plots saved successfully.")


def plot_all_figures(batch_size, epoch, history, y_true_labels, y_pred_labels, y_pred_probs, categories, result_out, model_name):
    """
    Plots Accuracy, Loss, Confusion Matrix, ROC Curve, and Accuracy vs. Recall plots.
    """
    # ✅ Vẽ Accuracy Plot
    plt.figure()
    plt.plot(history["train_accuracy"], label="Train Accuracy", linestyle="--", marker="o")
    plt.plot(history["val_accuracy"], label="Validation Accuracy", linestyle="-", marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(result_out, f"{model_name}_bs{batch_size}_ep{epoch}_accuracy_plot.png"))
    plt.close()

    # ✅ Vẽ Loss Plot
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss", linestyle="--", marker="o")
    plt.plot(history["val_loss"], label="Validation Loss", linestyle="-", marker="o")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(result_out, f"{model_name}_bs{batch_size}_ep{epoch}_loss_plot.png"))
    plt.close()

    print(f"✅ All plots saved to {result_out}")

    # 3. Confusion Matrix Plot with Float Numbers
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_confusion_matrix_normalized.png'))
    plt.close()

    # Encode the true labels to binary format
    label_encoder = LabelEncoder()
    y_true_binary = label_encoder.fit_transform(y_true_labels)

    # 4. ROC Curve Plot for each class in a one-vs-rest fashion
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10.colors  # Simplified colors
    line_styles = ['-', '--', '-.', ':']  # Updated line styles
    line_width = 1.5  # Reduced line thickness

    # Ensure y_true_labels and y_pred_labels are NumPy arrays and encode labels if they are not integers
    label_encoder = LabelEncoder()
    if isinstance(y_true_labels[0], str) or isinstance(y_true_labels[0], bool):
        y_true_labels = label_encoder.fit_transform(y_true_labels)
    else:
        y_true_labels = np.array(y_true_labels)

    if isinstance(y_pred_labels[0], str) or isinstance(y_pred_labels[0], bool):
        y_pred_labels = label_encoder.transform(y_pred_labels)
    else:
        y_pred_labels = np.array(y_pred_labels)

    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[1], linestyle=line_styles[0], linewidth=line_width, label=f'{categories[1]} (AUC = {roc_auc:.4f})')
        
        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, 0])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[0], linestyle=line_styles[1], linewidth=line_width, label=f'{categories[0]} (AUC = {roc_auc:.4f})')
        
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr, tpr,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (AUC = {roc_auc:.4f})'
            )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.0, label="Chance (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multiple Classes')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_roc_curve.png'))
    plt.close()

    # 5. Accuracy vs. Recall Plot
    report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)
    accuracy = [report[category]['precision'] for category in categories]
    recall = [report[category]['recall'] for category in categories]

    plt.figure()
    plt.plot(categories, accuracy, marker='o', linestyle='--', color='b', label='Accuracy')
    plt.plot(categories, recall, marker='o', linestyle='-', color='g', label='Recall')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.legend(loc='best')
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_accuracy_vs_recall.png'))
    plt.close()

    print(f"All plots saved to {result_out}")

    # 6. Precision-Recall Curves
    plt.figure(figsize=(10, 8))
    if len(categories) == 2:  # Binary classification case
        # Plotting for the positive class (1)
        y_true_binary = (y_true_labels == 1).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 1])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[1], linestyle=line_styles[0], linewidth=line_width, 
                 label=f'{categories[1]} (PR AUC = {pr_auc:.4f})')

        # Plotting for the negative class (0)
        y_true_binary = (y_true_labels == 0).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, 0])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[0], linestyle=line_styles[1], linewidth=line_width, 
                 label=f'{categories[0]} (PR AUC = {pr_auc:.4f})')
    else:  # Multi-class case
        for i, class_name in enumerate(categories):
            y_true_binary = (y_true_labels == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(
                recall, precision,
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=line_width,
                label=f'{class_name} (PR AUC = {pr_auc:.4f})'
            )

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(result_out, model_name + f'_bs{batch_size}_ep{epoch}_precision_recall_curve.png'))
    plt.close()

    
def load_features_without_smote(feature_paths, feature_type, categories, return_filenames=False):
    """
    Load multiple feature vectors from .npy files without applying SMOTE.
    
    Args:
        feature_paths (dict): Dictionary chứa đường dẫn đến feature của từng nhãn.
        feature_type (str): Loại đặc trưng cần tải.
        categories (list): Danh sách các nhãn.
        return_filenames (bool): Nếu True, trả về thêm danh sách tên file.
        
    Returns:
        np.ndarray: Feature matrix X (num_samples, num_features)
        np.ndarray: Encoded labels y
        (optional) list: Danh sách tên file tương ứng (nếu return_filenames=True)
    """
    all_features = []
    all_labels = []
    all_filenames = []  # Danh sách lưu tên file nếu cần
    label_encoder = LabelEncoder()

    for category in categories:
        if category not in feature_paths or feature_type not in feature_paths[category]:
            print(f"⚠️ Warning: No feature path for '{category}' and feature '{feature_type}'. Skipping.")
            continue

        folder_path = feature_paths[category][feature_type]
        if not os.path.isdir(folder_path):
            print(f"❌ Error: Feature folder '{folder_path}' does not exist. Skipping '{feature_type}'.")
            continue

        feature_vectors = []
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

        if not npy_files:
            print(f"⚠️ Warning: No .npy files found in '{folder_path}'.")

        # Sắp xếp danh sách file nếu cần đảm bảo thứ tự nhất định
        npy_files = sorted(npy_files)

        for filename in npy_files:
            file_path = os.path.join(folder_path, filename)
            try:
                feature = np.load(file_path)

                if feature.size == 0:
                    print(f"⚠️ Warning: '{filename}' is empty. Skipping.")
                    continue

                if feature.ndim == 1:
                    feature = feature.reshape(1, -1)  # Đảm bảo 2D shape (1, num_features)

                feature_vectors.append(feature)
                if return_filenames:
                    all_filenames.append(filename)
            except Exception as e:
                print(f"❌ Error loading '{file_path}': {e}")
                continue

        if len(feature_vectors) > 0:
            feature_matrix = np.vstack(feature_vectors)  # (num_samples, num_features)
            all_features.append(feature_matrix)
            num_samples = feature_matrix.shape[0]
            all_labels.extend([category] * num_samples)

    if len(all_features) == 0:
        print("⚠️ No valid features found. Returning empty arrays.")
        if return_filenames:
            return np.array([]), np.array([]), []
        else:
            return np.array([]), np.array([])

    X = np.vstack(all_features)  # Ghép tất cả mẫu lại
    y = np.array(all_labels)
    y_encoded = label_encoder.fit_transform(y)

    print(f"✅ Loaded {X.shape[0]} samples with feature type {feature_type}, feature size {X.shape[1]}.")
    if return_filenames:
        return X, y_encoded, all_filenames
    else:
        return X, y_encoded


class GNNClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def visualize_graph(graph_data, labels, save_path="graph_visualizations", sample_size=30):
    """
    Visualizes the graph structure using different techniques.

    Parameters:
    - graph_data: A PyG Data object containing the graph structure.
    - labels: Tensor containing node labels for color mapping.
    - save_path: Path to save the generated plots.
    - sample_size: Number of nodes to display in graph visualizations.
    """
    os.makedirs(save_path, exist_ok=True)

    # Handle empty graphs
    if graph_data.num_nodes == 0:
        print("⚠️ Empty graph! No visualizations generated.")
        return

    num_nodes = graph_data.num_nodes
    num_edges = graph_data.edge_index.shape[1]

    # Convert to NetworkX format
    G = nx.Graph()
    edge_index_np = graph_data.edge_index.t().cpu().numpy()
    G.add_edges_from(edge_index_np)
    G.add_nodes_from(range(num_nodes))  # Ensure isolated nodes are included

    # Sample nodes for better visualization
    sample_nodes = random.sample(range(num_nodes), min(sample_size, num_nodes))
    subgraph = G.subgraph(sample_nodes)
    pos = nx.kamada_kawai_layout(subgraph)

    # Ensure positions exist for all nodes
    for node in sample_nodes:
        if node not in pos:
            pos[node] = (0, 0)

    # Replace numeric labels with text labels if available
    label_dict = {i: labels[i].item() for i in sample_nodes}
    if isinstance(labels[0].item(), int) and hasattr(graph_data, 'category_labels'):
        label_dict = {i: graph_data.category_labels[labels[i].item()] for i in sample_nodes}

    # 1️⃣ **Raw Graph Structure**
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, pos, node_size=300, node_color="lightblue", with_labels=False)
    nx.draw_networkx_edges(subgraph, pos, edge_color="black", alpha=0.5, width=1.0)
    # plt.title("1️⃣ Raw Graph Structure (Sampled)")
    plt.savefig(os.path.join(save_path, "raw_graph_structure.png"))
    plt.close()

    # 2️⃣ **Graph with Labels**
    plt.figure(figsize=(8, 6))
    nx.draw(subgraph, pos, node_size=300, node_color="lightblue", edgecolors="black", with_labels=False)
    nx.draw_networkx_edges(subgraph, pos, edge_color="black", alpha=0.5, width=1.0)
    nx.draw_networkx_labels(subgraph, pos, labels=label_dict, font_size=8, font_color="black")
    # plt.title("2️⃣ Graph with Labels (Sampled)")
    plt.savefig(os.path.join(save_path, "graph_with_labels.png"))
    plt.close()

    # 3️⃣ **Graph with Clustering**
    plt.figure(figsize=(8, 6))
    color_map = plt.colormaps.get_cmap("tab10")
    node_colors = [color_map(labels[i] % 10) for i in sample_nodes]

    nx.draw(subgraph, pos, node_color=node_colors, node_size=300, edgecolors="black", with_labels=False)
    nx.draw_networkx_edges(subgraph, pos, edge_color="black", alpha=0.5, width=1.0)
    nx.draw_networkx_labels(subgraph, pos, labels=label_dict, font_size=8, font_color="black")
    # plt.title("3️⃣ Graph with Clustering (Sampled)")
    plt.savefig(os.path.join(save_path, "graph_with_clustering.png"))
    plt.close()

    # 4️⃣ **PCA Projection of Node Embeddings**
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        node_embeddings = graph_data.x.cpu().numpy()
        pca_embeddings = PCA(n_components=2).fit_transform(node_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels.cpu().numpy(), cmap="tab10", s=30)
        for i, (x, y) in enumerate(pca_embeddings[:, :2]):
            plt.text(x, y, str(label_dict.get(i, i)), fontsize=6, ha="right")

        # plt.title("4️⃣ PCA Projection of Embeddings")
        plt.savefig(os.path.join(save_path, "pca_projection.png"))
        plt.close()

        # 5️⃣ **t-SNE Projection of Node Embeddings**
        perplexity = min(30, num_nodes // 10)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(pca_embeddings)

        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels.cpu().numpy(), cmap="tab10", s=30)
        for i, (x, y) in enumerate(tsne_embeddings):
            plt.text(x, y, str(label_dict.get(i, i)), fontsize=6, ha="right")

        # plt.title("5️⃣ t-SNE Projection of Embeddings")
        plt.savefig(os.path.join(save_path, "tsne_projection.png"))
        plt.close()

    print(f"✅ Graph visualizations saved in {save_path}")


def run_experiment(train_paths, val_paths, epoch_values, batch_size_list,
                   metric_collection, result_folder, categories, feature_types,
                   metadata_df=None, one_hot_encoder=None, age_scaler=None,
                   visualize=True):
    from sklearn.model_selection import RepeatedStratifiedKFold
    import pickle

    print("\n📅 Loading training features...")
    X_train_list, y_train, train_file_names = [], None, []

    for ft in feature_types:
        X_train_ft, y_train_ft, fnames = load_features_without_smote(train_paths, ft, categories, return_filenames=True)
        if y_train is None:
            y_train = y_train_ft
        X_train_list.append(X_train_ft)
        train_file_names = fnames

    X_train_img = np.hstack(X_train_list)

    if metadata_df is not None and one_hot_encoder is not None and age_scaler is not None:
        X_train_clinical = extract_clinical_features_from_list(train_file_names, metadata_df, one_hot_encoder, age_scaler)
        X_train = np.hstack([X_train_img, X_train_clinical])
    else:
        X_train = X_train_img

    if val_paths and any(val_paths.values()):
        print("📅 Loading validation features...")
        X_val_list, y_val, val_file_names = [], None, []

        for ft in feature_types:
            X_val_ft, y_val_ft, fnames = load_features_without_smote(val_paths, ft, categories, return_filenames=True)
            if y_val is None:
                y_val = y_val_ft
            X_val_list.append(X_val_ft)
            val_file_names = fnames

        X_val_img = np.hstack(X_val_list)

        if metadata_df is not None and one_hot_encoder is not None and age_scaler is not None:
            X_val_clinical = extract_clinical_features_from_list(val_file_names, metadata_df, one_hot_encoder, age_scaler)
            X_val = np.hstack([X_val_img, X_val_clinical])
        else:
            X_val = X_val_img

        X_all = np.vstack((X_train, X_val))
        y_all = np.hstack((y_train, y_val))
    else:
        print("⚠️ No validation set provided. Using only training set for cross-validation.")
        X_all = X_train
        y_all = y_train

    X_all, y_all = augment_all_classes_to_balance(X_all, y_all, noise_std=0.01)
    mask_valid = ~np.isnan(X_all).any(axis=1)
    X_all = X_all[mask_valid]
    y_all = y_all[mask_valid]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)

    dist_plot_path = os.path.join(result_folder, "label_distribution_after_augmentation.png")
    plot_label_distribution(y_encoded, label_encoder, dist_plot_path)

    model_name = "GNN_deep_handcrafted_clinical"
    model_result_out = os.path.join(result_folder, model_name)
    os.makedirs(model_result_out, exist_ok=True)

    best_score = -1
    best_model_state = None
    best_model_info = {}
    all_histories = {}

    for batch_size in batch_size_list:
        for epoch in epoch_values:
            rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            fold = 1
            fold_metrics = []

            for train_idx, test_idx in rkf.split(X_all, y_encoded):
                print(f"\n🔁 Fold {fold}/15")
                X_all_fold, _ = normalize_data(np.nan_to_num(X_all), np.nan_to_num(X_all))

                input_dim = X_all_fold.shape[1]
                num_classes = len(np.unique(y_encoded))

                gnn_model = GNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
                optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)

                epoch_result_out = os.path.join(model_result_out, f'batch_size_{batch_size}', f'epoch_{epoch}_fold_{fold}')
                os.makedirs(epoch_result_out, exist_ok=True)

                data = create_graph(X_all_fold, y_encoded, train_idx=train_idx, test_idx=test_idx, k=5, use_mask=True)
                visualize_graph(data, torch.tensor(y_encoded), save_path=f"{epoch_result_out}/graph")

                start_time = time.time()
                gnn_model, history = train_gnn_model(gnn_model, data, optimizer, epoch, epoch_result_out)
                end_time = time.time()

                all_histories.setdefault(model_name, []).append({
                    "batch_size": batch_size,
                    "epoch": epoch,
                    "history": history
                })

                gnn_model.eval()
                with torch.no_grad():
                    output = gnn_model(data)
                    test_mask = data.test_mask
                    y_pred_probs = F.softmax(output[test_mask], dim=1).cpu().numpy()

                y_pred_labels = np.argmax(y_pred_probs, axis=1)
                y_true_labels = y_encoded[test_mask.cpu().numpy()]

                best_epoch = np.argmax(history["val_accuracy"]) + 1
                best_val_accuracy = history["val_accuracy"][best_epoch - 1]

                if num_classes == 2:
                    auc_value = roc_auc_score(y_true_labels, y_pred_probs[:, 1])
                else:
                    auc_value = roc_auc_score(y_true_labels, y_pred_probs, multi_class='ovr', average='macro')

                test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
                precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
                recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
                f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

                report = classification_report(y_true_labels, y_pred_labels, target_names=categories, output_dict=True)

                fold_metrics.append({
                    "F1": f1,
                    "Recall": recall,
                    "Precision": precision,
                    "Accuracy": test_accuracy,
                    "Macro F1": report["macro avg"]["f1-score"],
                    "Macro Recall": report["macro avg"]["recall"],
                    "Macro Precision": report["macro avg"]["precision"],
                    "Macro AUC": auc_value,
                    "Best Epoch": best_epoch,
                    "Best Validation Accuracy": best_val_accuracy,
                    "Time Taken": end_time - start_time,
                })

                fold_report_path = os.path.join(epoch_result_out, f"fold_{fold}_metrics.txt")
                with open(fold_report_path, "w") as f:
                    f.write("Fold Evaluation Report\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Fold Index     : {fold}\n")
                    f.write(f"Epoch          : {epoch}\n")
                    f.write(f"Batch Size     : {batch_size}\n")
                    f.write(f"Best Epoch     : {best_epoch}\n")
                    f.write(f"Best Val Acc   : {best_val_accuracy:.4f}\n")
                    f.write(f"Accuracy       : {test_accuracy:.4f}\n")
                    f.write(f"Precision      : {precision:.4f}\n")
                    f.write(f"Recall         : {recall:.4f}\n")
                    f.write(f"F1 Score       : {f1:.4f}\n")
                    f.write(f"Macro F1       : {report['macro avg']['f1-score']:.4f}\n")
                    f.write(f"Macro Recall   : {report['macro avg']['recall']:.4f}\n")
                    f.write(f"Macro Precision: {report['macro avg']['precision']:.4f}\n")
                    f.write(f"Macro AUC      : {auc_value:.4f}\n")
                    f.write(f"Time Taken (s) : {end_time - start_time:.2f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(classification_report(y_true_labels, y_pred_labels, target_names=categories))

                plot_all_figures(
                    batch_size=batch_size,
                    epoch=epoch,
                    history=history,
                    y_true_labels=y_true_labels,
                    y_pred_labels=y_pred_labels,
                    y_pred_probs=y_pred_probs,
                    categories=categories,
                    result_out=epoch_result_out,
                    model_name=model_name
                )

                score = report["macro avg"]["f1-score"] + report["macro avg"]["recall"] + report["macro avg"]["precision"] - (np.std([f1]) + np.std([recall]))
                if score > best_score:
                    best_score = score
                    best_model_state = gnn_model.state_dict()
                    best_model_info = {
                        "score": best_score,
                        "fold": fold,
                        "epoch": epoch,
                        "batch_size": batch_size,
                        "label_encoder": label_encoder,
                        "macro_f1": report["macro avg"]["f1-score"],
                        "macro_recall": report["macro avg"]["recall"],
                        "macro_precision": report["macro avg"]["precision"],
                        "macro_auc": auc_value,
                        "model_name": model_name
                    }

                fold += 1

            if fold_metrics:
                df_fold = pd.DataFrame(fold_metrics)
                metric_summary = {
                    "Model": model_name,
                    "Batch Size": batch_size,
                    "Epoch": epoch,
                    "Best Epoch Mean": df_fold["Best Epoch"].mean(),
                    "Best Epoch Std": df_fold["Best Epoch"].std(),
                    "Best Val Acc Mean": df_fold["Best Validation Accuracy"].mean(),
                    "Best Val Acc Std": df_fold["Best Validation Accuracy"].std(),
                    "Test Accuracy": df_fold["Accuracy"].mean(),
                    "Test Accuracy Std": df_fold["Accuracy"].std(),
                    "Precision": df_fold["Precision"].mean(),
                    "Precision Std": df_fold["Precision"].std(),
                    "Recall": df_fold["Recall"].mean(),
                    "Recall Std": df_fold["Recall"].std(),
                    "F1 Score": df_fold["F1"].mean(),
                    "F1 Score Std": df_fold["F1"].std(),
                    "Macro F1": df_fold["Macro F1"].mean(),
                    "Macro F1 Std": df_fold["Macro F1"].std(),
                    "Macro Precision": df_fold["Macro Precision"].mean(),
                    "Macro Precision Std": df_fold["Macro Precision"].std(),
                    "Macro Recall": df_fold["Macro Recall"].mean(),
                    "Macro Recall Std": df_fold["Macro Recall"].std(),
                    "Macro AUC": df_fold["Macro AUC"].mean(),
                    "Macro AUC Std": df_fold["Macro AUC"].std(),
                    "Training Time (s)": df_fold["Time Taken"].mean(),
                    "Training Time Std (s)": df_fold["Time Taken"].std(),
                }

                metric_collection.append(metric_summary)

    if best_model_state is not None:
        final_model_path = os.path.join(model_result_out, "best_overall_model.pth")
        torch.save(best_model_state, final_model_path)
        with open(os.path.join(model_result_out, "best_label_encoder.pkl"), "wb") as f:
            pickle.dump(best_model_info["label_encoder"], f)

        with open(os.path.join(model_result_out, "best_model_info.txt"), "w") as f:
            for k, v in best_model_info.items():
                f.write(f"{k}: {v}\n")

    pd.DataFrame(metric_collection).to_csv(os.path.join(result_folder, 'performance_metrics.csv'), index=False)
    if visualize:
        plot_combined_metrics(metric_collection, result_folder)
        plot_epoch_based_metrics(all_histories, result_folder)

    print("✅ Cross-validated experiment completed.")
    return all_histories, metric_collection



# Function to dynamically generate train/test/val paths
def generate_paths(feature_dir, dataset_type, feature_types, categories):
    """
    Tạo dictionary đường dẫn đặc trưng theo nhãn và loại đặc trưng.
    Hỗ trợ cả tập dữ liệu không có nhãn (unlabeled) trong tập test.

    Args:
        feature_dir (str): Thư mục gốc chứa đặc trưng.
        dataset_type (str): 'train', 'val', hoặc 'test'
        feature_types (list): Danh sách các đặc trưng (ví dụ: ['glcm', 'hsv_histograms'])
        categories (list): Danh sách các nhãn (ví dụ: ['MEL', 'NV', ...])

    Returns:
        dict: paths[category][feature_type] = đường dẫn thư mục
    """
    paths = {}
    print(f"📁 Checking paths for dataset type: {dataset_type}")

    for category in categories:
        paths[category] = {}
        for feature_type in feature_types:
            feature_path = os.path.join(feature_dir, dataset_type, feature_type, category)
            print(f"🔍 {feature_path}")
            if os.path.exists(feature_path) and os.path.isdir(feature_path):
                paths[category][feature_type] = feature_path
            else:
                print(f"❌ Not Found: {feature_path}")

    # ✅ Nếu là tập test → kiểm tra thêm thư mục 'unlabeled'
    if dataset_type == 'test':
        unlabeled_category = 'unlabeled'
        paths[unlabeled_category] = {}
        for feature_type in feature_types:
            unlabeled_path = os.path.join(feature_dir, dataset_type, feature_type, unlabeled_category)
            print(f"🔍 (Unlabeled) {unlabeled_path}")
            if os.path.exists(unlabeled_path) and os.path.isdir(unlabeled_path):
                paths[unlabeled_category][feature_type] = unlabeled_path
            else:
                print(f"❌ Not Found: {unlabeled_path}")

    return paths


def predict_labeled_test_set(model_path, label_encoder_path, test_paths, feature_types, result_dir, categories,
                              metadata_df=None, one_hot_encoder=None, age_scaler=None):
    import pickle

    print(f"\n🔍 [TEST] Evaluating fused model on test set with features: {feature_types}")
    X_test_list, y_test, file_names = [], None, None
    first_feature = True
    all_valid = True

    for feature_type in feature_types:
        if first_feature:
            X_part, y_part, file_names_part = load_features_without_smote(test_paths, feature_type, categories, return_filenames=True)
            if X_part.size == 0:
                print(f"⚠️ No valid features found for feature '{feature_type}'. Skipping.")
                all_valid = False
                continue
            file_names = file_names_part
            y_test = y_part
            first_feature = False
        else:
            X_part, _, _ = load_features_without_smote(test_paths, feature_type, categories, return_filenames=True)
            if X_part.size == 0:
                print(f"⚠️ No valid features found for feature '{feature_type}'. Skipping.")
                all_valid = False
                continue

        X_part = np.nan_to_num(X_part, nan=np.nanmean(X_part))
        X_test_list.append(X_part)

    if not X_test_list or not all_valid:
        print("❌ No valid test features found. Skipping evaluation.")
        return

    X_img = np.hstack(X_test_list)

    # ✅ Ghép clinical metadata (multi-modal)
    if metadata_df is not None and one_hot_encoder and age_scaler and file_names:
        X_clinical = extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler)
        X_test_combined = np.hstack([X_img, X_clinical])
    else:
        X_test_combined = X_img

    X_test_combined, _ = normalize_data(X_test_combined, X_test_combined)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(np.unique(y_test_encoded))

    input_dim = X_test_combined.shape[1]
    gnn_model = GNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=num_classes)
    gnn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    gnn_model.eval()

    test_graph = create_graph(X_test_combined, y_test_encoded)

    with torch.no_grad():
        output = gnn_model(test_graph)
        probs = F.softmax(output, dim=1).cpu().numpy()

    preds = np.argmax(probs, axis=1)

    # 🎯 Evaluation
    cm = confusion_matrix(y_test_encoded, preds)
    test_acc = accuracy_score(y_test_encoded, preds)
    precision = precision_score(y_test_encoded, preds, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, preds, average='weighted', zero_division=0)

    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if cm.shape[0] > 1 else 0
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape[0] > 1 else 0

    report_txt = classification_report(y_test_encoded, preds, target_names=categories)
    report_dict = classification_report(y_test_encoded, preds, target_names=categories, output_dict=True)

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, "final_test_classification_report.txt"), "w") as f:
        f.write(report_txt)

    with open(os.path.join(result_dir, "final_test_evaluation_metrics.txt"), "w") as f:
        f.write("Final Test Set Evaluation Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Accuracy       : {test_acc:.4f}\n")
        f.write(f"Precision      : {precision:.4f}\n")
        f.write(f"Recall         : {recall:.4f}\n")
        f.write(f"F1 Score       : {f1:.4f}\n")
        f.write(f"Sensitivity    : {sensitivity:.4f}\n")
        f.write(f"Specificity    : {specificity:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    # ✅ Lưu dự đoán per file
    true_labels_str = label_encoder.inverse_transform(y_test_encoded)
    predicted_labels_str = label_encoder.inverse_transform(preds)

    if file_names and len(file_names) == len(true_labels_str):
        df_result = pd.DataFrame({
            "Filename": file_names,
            "True Label": true_labels_str,
            "Predicted Label": predicted_labels_str
        })
        df_result.to_csv(os.path.join(result_dir, "test_per_file_predictions.csv"), index=False)
        print("✅ Test predictions saved.")


def predict_unlabeled_test_set(model_path, label_encoder_path, test_paths, feature_types, result_dir,
                                metadata_df=None, one_hot_encoder=None, age_scaler=None):
    import pickle
    import re

    print("\n🧪 Predicting unlabeled test set...")
    X_test_list, file_names = [], None
    all_valid = True

    for ft in feature_types:
        if 'unlabeled' not in test_paths:
            continue
        folder = test_paths['unlabeled'].get(ft)
        if not folder or not os.path.exists(folder):
            print(f"⚠️ Feature '{ft}' not found for unlabeled test set. Skipping.")
            all_valid = False
            continue

        feature_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
        if not feature_files:
            print(f"⚠️ No .npy files found in folder '{folder}'. Skipping feature '{ft}'.")
            all_valid = False
            continue

        vectors = [np.load(f).reshape(1, -1) for f in feature_files if os.path.getsize(f) > 0]

        if vectors:
            X_ft = np.vstack(vectors)
            X_test_list.append(X_ft)

            if file_names is None:
                file_names = [os.path.basename(f).replace(".npy", "") for f in feature_files]
        else:
            print(f"⚠️ All vectors in '{folder}' were empty or invalid.")
            all_valid = False

    if not X_test_list or not all_valid:
        print("❌ No valid features found for prediction. Skipping.")
        return

    X_img = np.hstack(X_test_list)

    # ✅ Ghép clinical metadata
    if metadata_df is not None and one_hot_encoder and age_scaler and file_names:
        X_clinical = extract_clinical_features_from_list(file_names, metadata_df, one_hot_encoder, age_scaler)
        X_combined = np.hstack([X_img, X_clinical])
    else:
        X_combined = X_img

    X_combined, _ = normalize_data(X_combined, X_combined)

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    input_dim = X_combined.shape[1]
    output_dim = len(label_encoder.classes_)
    model = GNNClassifier(input_dim=input_dim, hidden_dim=64, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_labels = np.zeros(len(X_combined), dtype=int)
    graph = create_graph(X_combined, labels=dummy_labels)

    with torch.no_grad():
        output = model(graph)
        probs = F.softmax(output, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        predicted_labels = label_encoder.inverse_transform(preds)

    os.makedirs(result_dir, exist_ok=True)
    df_result = pd.DataFrame({
        "Filename": file_names,
        "Predicted Label": predicted_labels
    })
    df_result.to_csv(os.path.join(result_dir, "unlabeled_predictions.csv"), index=False)
    print(f"✅ Predictions saved to {result_dir}")



def main():
    """Tổng quát hóa pipeline GNN: xử lý mọi trường hợp thiếu val/test."""

    # 1️⃣ Định nghĩa thư mục dữ liệu
    base_dir = os.getcwd()
    home_dir = os.path.join(base_dir, 'data9') 
    train_metadata_path = os.path.join(home_dir, "ISIC_2020_Train_Metadata.csv")
    test_metadata_path = os.path.join(home_dir, "ISIC_2020_Test_Metadata.csv")

    feature_dir = os.path.join(home_dir, 'data9_SOTA_and_handcrafts_and_BlookNet_optimal_entropy_features_v3')
    result_folder = os.path.join(home_dir, 'training_data9_Multi_modal_GNN_Skin_Lesion_Classification_v1')
    os.makedirs(result_folder, exist_ok=True)

    # 2️⃣ Định nghĩa nhãn & đặc trưng
    categories = ['benign', 'malignant']
    feature_types = [
        # "hog_features", 
        # "lbp_features", 
        "color_histograms_features", 
        "hsv_histograms_features", # selected
        # "gabor_features", 
        # "glcm_features", 
        # "wavelet_features", 
        "fractal_features", 
        # "edge_features",
        # "color_correlation_features", 
        # "vgg16_features", 
        "vgg19_features", 
        # "resnet50_features",
        # "resnet101_features",
        # "resnet152_features", 
        # "inceptionv3_features", 
        "mobilenet_features", # selected
        # "efficientnetb0_features", 
        # "efficientnetb7_features", 
        "densenet121_features", 
        # "densenet169_features", # selected
        # "densenet201_features", # selected
        # "vit_b16_features"
    ]

    # 3️⃣ Sinh đường dẫn đặc trưng
    train_paths = generate_paths(feature_dir, "train", feature_types, categories)
    val_paths = generate_paths(feature_dir, "val", feature_types, categories)
    test_paths = generate_paths(feature_dir, "test", feature_types, categories)

    # ⚠️ 4️⃣ Nếu val_paths rỗng → bỏ qua val, huấn luyện cross-validation
    if not any(val_paths.values()):
        print("⚠️ Validation set not found. Proceeding with only training set and cross-validation.")
        val_paths = None

    # 5️⃣ Siêu tham số
    batch_size_list = [32]
    epoch_values = [100]
    metric_collection = []

    # 6️⃣ Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # 7️⃣ Cấu hình
    print("\n📌 Experiment Configuration:")
    print(f"🔹 Categories: {categories}")
    print(f"🔹 Feature types: {feature_types}")
    print(f"🔹 Batch sizes: {batch_size_list}")
    print(f"🔹 Epoch values: {epoch_values}")
    print(f"🔹 Feature Dir: {feature_dir}")
    print(f"🔹 Result Dir : {result_folder}")

    # 8️⃣ Huấn luyện mô hình
    print("\n🚀 Running GNN experiments with cross-validation...")
    train_metadata_df = pd.read_csv(train_metadata_path)
    one_hot_encoder, age_scaler = prepare_clinical_encoder(train_metadata_df)

    all_histories, metric_collection = run_experiment(
        train_paths=train_paths,
        val_paths=val_paths,
        epoch_values=epoch_values,
        batch_size_list=batch_size_list,
        metric_collection=metric_collection,
        result_folder=result_folder,
        categories=categories,
        feature_types=feature_types,
        metadata_df=train_metadata_df,
        one_hot_encoder=one_hot_encoder,
        age_scaler=age_scaler,
        visualize=True
    )



    print("✅ Experiments completed.")

    # 9️⃣ Lưu kết quả
    metrics_file = os.path.join(result_folder, 'overall_performance_metrics.csv')
    pd.DataFrame(metric_collection).to_csv(metrics_file, index=False)
    print(f"📊 Metrics saved: {metrics_file}")

    # 🔟 Vẽ biểu đồ
    print("\n📈 Drawing plots...")
    plot_combined_metrics(metric_collection, result_folder)
    plot_epoch_based_metrics(all_histories, result_folder)

    # 🔟🔟 Đánh giá test (nếu có)
    model_name = "GNN_deep_handcrafted_clinical"
    model_dir = os.path.join(result_folder, model_name)
    model_path = os.path.join(model_dir, "best_overall_model.pth")
    label_path = os.path.join(model_dir, "best_label_encoder.pkl")

    if os.path.exists(model_path) and os.path.exists(label_path):
        # 📖 Đọc metadata cho test
        test_metadata_df = pd.read_csv(test_metadata_path)

        # 1️⃣ Đánh giá test có nhãn
        if any(cat in test_paths for cat in categories):
            print("\n🔍 Evaluating on labeled test set...")
            predict_labeled_test_set(
                model_path=model_path,
                label_encoder_path=label_path,
                test_paths=test_paths,
                feature_types=feature_types,
                result_dir=model_dir,
                categories=categories,
                metadata_df=test_metadata_df,
                one_hot_encoder=one_hot_encoder,
                age_scaler=age_scaler
            )

        # 2️⃣ Dự đoán test không nhãn (unlabeled)
        if 'unlabeled' in test_paths:
            print("\n🧪 Predicting on unlabeled test set...")
            predict_unlabeled_test_set(
                model_path=model_path,
                label_encoder_path=label_path,
                test_paths=test_paths,
                feature_types=feature_types,
                result_dir=model_dir,
                metadata_df=test_metadata_df,
                one_hot_encoder=one_hot_encoder,
                age_scaler=age_scaler
            )
    else:
        print("⚠️ Model or LabelEncoder not found. Skipping test evaluation and prediction.")


# 9️⃣ Chạy chương trình
if __name__ == "__main__":
    main()
