
import numpy as np
import torch
import os
from copy import deepcopy
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns
import umap.umap_ as umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import random

from model.pointnet import PointNet
from dataset import MyDataset_test
from bag.utils.generic_utils import segment_fn
from utils import data_reading, pc_normalize


# Setting environment variables to ensure deterministic CUDA operation
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# set fix all seed
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Use with multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Correctly set up the deterministic algorithm
    torch.use_deterministic_algorithms(True)

set_seed(42)

class PCL_Encoder(torch.nn.Module):
    def __init__(self, input_dim=1024, output_dim=8):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def compute_similarity(features, prototypes, tau=0.5):
    # Remove unnecessary type conversion (assuming input is already float32)
    similarity = torch.nn.functional.cosine_similarity(
        features[:, None], prototypes[None], dim=-1)
    return torch.nn.functional.softmax(similarity / tau, dim=1)


def train_pcl(encoder, X, n_clusters=2, epochs=10, tau=0.5, lr=1e-4):
    encoder.to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    encoder.train()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        embeddings = encoder(X_tensor)
        embeddings_np = embeddings.detach().cpu().numpy()

        # KMeans fixed initialization
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(embeddings_np)
        prototypes = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

        sim = compute_similarity(embeddings, prototypes, tau)
        loss = -torch.mean(torch.log(sim.max(dim=1).values + 1e-10))  # 添加极小值防止log(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item():.4f}")

    with torch.no_grad():
        return encoder(X_tensor).cpu().numpy()



def segment_extraction(pc, mask, influ_name, segment_num, segment_fn, weight_threshold):
    segments, fudged_data = segment_fn(pc, segment_num)  # What is stored is the mean of each fragment, keeping the same size as the original data
    segment_HC_attributes = [] # Store fragments that tend to be HC
    segment_PD_attributes = [] # Store fragments that tend to be PD
    # Each segment is intercepted and divided into two groups, HC and PD, according to the weight value.
    n_segments = np.unique(segments)
    for idx in n_segments:
        indices = np.where(segments == idx)[0]  # Get all indexes equal to the value, that is, the index corresponding to the fragment
        weight = np.unique(mask[indices])
        assert len(weight) == 1, f"Error: Expected number of weights in one segment to be 1, but got {len(weight)}"
        reversed_data = deepcopy(pc)
        segment_data = reversed_data[indices][:, influ_name]
        segment_data = pc_normalize(segment_data)
        # According to the attribution value, the corresponding hand-drawn fragment is stored in the corresponding attribute fragment list
        if weight > weight_threshold and segment_data.shape[0] > 10:
            segment_PD_attributes.append(segment_data)
        elif weight < -weight_threshold and segment_data.shape[0] > 10:
            segment_HC_attributes.append(segment_data)
    return segment_HC_attributes, segment_PD_attributes




def segment_predict(segment_dataset, model_name, label_type):
    if len(segment_dataset) == 0:  # Avoid empty data entering DataLoader
        return None

    features_1024_list = []
    test_dataset = MyDataset_test(dataset=segment_dataset, name=label_type, transform=None)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

    for data in test_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        with torch.no_grad():
            if model_name == 'PointNet':
                _, _, feature_1024 = model(inputs)  # Only 1024-dimensional features (latent space) are obtained

            feature_np = feature_1024.cpu().detach().numpy().squeeze(0)  # ensure shape (1024,)
            features_1024_list.append(feature_np)

    if len(features_1024_list) == 0:
        return None

    return np.array(features_1024_list)  # return (N, 1024)




if __name__=='__main__':

    # weight_thresholds = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
    # n_neighbors_list = [5, 20, 30, 50, 100, 105, 110, 120, 125, 130, 135, 140, 145, 150, 155, 160, 180, 200]
    n_clusters_list = [2]
    weight_thresholds = [0.0001]  # Attribution Threshold
    n_neighbors_list = [300]  # UMAP Neighbors

    data_pattern = 1

    neighbor_num = 200
    explainer_type = 'xgb'
    influ_name = [0, 1, 2]  # See which features affect the model results
    segment_num = 10
    model_name = 'PointNet'
    dataset = 'ParkinsonHW'
    stride_size = 8

    out_path = os.path.join('./data', dataset, f'result_K_{neighbor_num}_{explainer_type}')
    if not os.path.exists(out_path):
        os.mkdir(out_path)


    best_sil_score = -1
    best_db_index = 100
    best_ch_index = -1
    best_params = None
    window_size = 256 if data_pattern == 0 else 512

    for weight_threshold in weight_thresholds:
        segment_HC = []
        segment_PD = []

        for fold_d in ['fold_1', 'fold_2', 'fold_3']:
            if data_pattern == 0:
                if fold_d == 'fold_1':
                    time_date = '2024_11_20_17_05_26'
                elif fold_d == 'fold_2':
                    time_date = '2024_11_20_17_22_30'
                elif fold_d == 'fold_3':
                    time_date = '2024_11_20_17_46_32'
            elif data_pattern == 1:
                if fold_d == 'fold_1':
                    time_date = '2024_11_19_18_32_47'
                elif fold_d == 'fold_2':
                    time_date = '2024_11_19_20_19_30'
                elif fold_d == 'fold_3':
                    time_date = '2024_11_19_20_39_53'

            model_path = os.path.join('./log_dir', time_date, 'checkpoints/best_model/PointNet_cls.pth')
            file_path = os.path.join('./data', dataset, fold_d, 'test_names.txt')

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            if model_name == 'PointNet':
                model = PointNet()
            model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device)['state_dict'])
            model.eval()

            file_path = os.path.join('./data', dataset, fold_d, 'test_names.txt')
            with open(file_path, 'r') as f:
                for line in f.readlines():

                    label_type = line.strip().split('/')[0]
                    file_name = line.strip().split('/')[1].split('.')[0]
                    json_file_path = os.path.join('./data', dataset, 'raw_data', line.strip())
                    temp_data, L = data_reading(json_file_path, dataset, data_pattern)
                    # 0-x,1-y,2-z,3-p,4-g,5-t,6-v,7-acc,8-jerk,9-dx,10-dy,11-da,12-dl,13-dp,14-dt,15-radius,16-angle,17-curvature,18-idx
                    if L > window_size:
                        pc = temp_data[:, [0, 1, 3]] if data_pattern == 0 else temp_data[:, [0, 1, 15]]
                        mask = np.load(os.path.join(out_path, file_name + '-' + str(data_pattern), 'weight.npy')).reshape(-1)

                        segment_HC_attributes, segment_PD_attributes = segment_extraction(
                            pc=pc,
                            mask=mask,
                            influ_name=influ_name,
                            segment_num=segment_num,
                            segment_fn=segment_fn,
                            weight_threshold=weight_threshold
                        )

                        if len(segment_HC_attributes) > 0:
                            segment_HC_features = segment_predict(segment_HC_attributes, model_name, label_type)
                            if segment_HC_features is not None:
                                segment_HC.extend(segment_HC_features)

                        if len(segment_PD_attributes) > 0:
                            segment_PD_features = segment_predict(segment_PD_attributes, model_name, label_type)
                            if segment_PD_features is not None:
                                segment_PD.extend(segment_PD_features)

        # Make sure segment_HC and segment_PD are NumPy arrays
        segment_HC = np.array(segment_HC)  # HC category 1024-dimensional features
        segment_PD = np.array(segment_PD)  # PD category 1024-dimensional features
        # Avoid np.vstack() errors
        if segment_HC.shape[0] == 0 or segment_PD.shape[0] == 0:
            continue

        # combine HC and PD
        X = np.concatenate([segment_HC, segment_PD], axis=0)  # shape: (N, 1024)
        y = np.array([0] * len(segment_HC) + [1] * len(segment_PD))

        for n_clusters in n_clusters_list:
            pcl_encoder = PCL_Encoder(input_dim=1024, output_dim=8)
            set_seed(42)
            X_low = train_pcl(encoder=pcl_encoder, X=X, n_clusters=n_clusters, epochs=300, tau=0.5, lr=1e-4)
            set_seed(42)

            for n_neighbors in n_neighbors_list:
                # Dimensionality reduction using UMAP
                reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, n_jobs=1, metric='cosine')
                embedding = reducer.fit_transform(X_low)  # Unsupervised Clustering

                # reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42, n_jobs=1, metric='cosine')
                # embedding = reducer.fit_transform(X_low, y=y)  # Supervised Clustering

                # Calculate the silhouette coefficient
                sil_score = silhouette_score(embedding, y)
                print(
                    f"##########---------- Silhouette Score (weight_threshold={weight_threshold}, n_neighbors={n_neighbors}, n_cluster={n_clusters}): {sil_score:.4f}")
                # DB metric
                db_index = davies_bouldin_score(embedding, y)
                print(
                    f"##########---------- Davies-Bouldin Index (weight_threshold={weight_threshold}, n_neighbors={n_neighbors}): {db_index:.4f}")
                # CH metric
                ch_index = calinski_harabasz_score(embedding, y)
                print(
                    f"##########---------- Calinski-Harabasz Index (weight_threshold={weight_threshold}, n_neighbors={n_neighbors}): {ch_index:.4f}\n")

                # plot
                plt.figure(figsize=(8, 6))
                plt.scatter(embedding[y == 0, 0], embedding[y == 0, 1], label="HC", alpha=0.7, s=5, color="blue")
                plt.scatter(embedding[y == 1, 0], embedding[y == 1, 1], label="PD", alpha=0.7, s=5, color="red")
                plt.legend()
                plt.title("UMAP Visualization of Handwriting Segments (HC vs PD)")
                plt.xlabel("UMAP Component 1")
                plt.ylabel("UMAP Component 2")
                plt.show()

                if sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_db_index = db_index
                    best_ch_index = ch_index
                    best_params = (weight_threshold, n_neighbors, n_clusters)

    print(
        f"Best Silhouette Score: {best_sil_score:.4f}, Best DB Score: {best_db_index:.4f}, Best CH Score: {best_ch_index:.4f} with weight_threshold={best_params[0]} and n_neighbors={best_params[1]} and n_clusters={best_params[2]} \n")





















