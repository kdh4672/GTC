import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.autograd import Variable
from sklearn.cluster import KMeans, AgglomerativeClustering
import torch
# import faiss
import pandas as pd

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def acc(y_true, y_pred, num_cluster):

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    w = np.zeros((num_cluster, num_cluster))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    ind = linear_sum_assignment(w.max() - w)
    accuracy = 0.0
    for i in ind[0]:
        accuracy = accuracy + w[i, ind[1][i]]
    return accuracy / y_pred.size


def Kmeans_model_evaluation(model, data_loader, num_cluster):
    model.eval()
    with torch.no_grad():
        dataset_size = len(data_loader.dataset)
        datas = np.zeros([dataset_size, 512])
        label_true = np.zeros(dataset_size)
        ii = 0
        for x, target in data_loader:
            x = Variable(x).cuda()
            _, word_emgedding = model(x)
            u = word_emgedding.cpu()
            datas[ii * data_loader.batch_size:(ii + 1) *
                  data_loader.batch_size, :] = u.data.numpy()
            label_true[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(datas)

        label_pred = kmeans.labels_
        ACC = acc(label_true, label_pred, num_cluster)
        print('ACC', ACC)
        print('NMI', nmi(label_true, label_pred))

    return ACC


def Kmeans_model_evaluation_text_representation(model, clip_model, pre_sentence_embedding, post_sentence_embedding, data_loader, num_cluster):
    model.eval()
    dataset_size = len(data_loader.dataset)
    datas = np.zeros([dataset_size, 768])
    label_true = np.zeros(dataset_size)
    ii = 0
    for x, target in data_loader:
        b = x.shape[0]
        x = Variable(x).cuda()
        word_embedding, _ = model(x)
        pre = pre_sentence_embedding.expand(b, -1, -1)
        post = post_sentence_embedding.expand(b, -1, -1)  # b,72,512
        total_embedding = torch.cat((pre, word_embedding, post), dim=1)
        text_representation = clip_model.encode_text(total_embedding)
        u = text_representation.cpu()
        datas[ii * data_loader.batch_size:(ii + 1) *
              data_loader.batch_size, :] = u.data.numpy()
        label_true[ii * data_loader.batch_size:(ii + 1)
                   * data_loader.batch_size] = target.numpy()
        ii = ii + 1
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(datas)

    label_pred = kmeans.labels_
    ACC = acc(label_true, label_pred, num_cluster)
    print('ACC', ACC)
    print('NMI', nmi(label_true, label_pred))

    return ACC


def base_kmeans_model_evaluation(model, data_loader, num_cluster):
    model.eval()
    with torch.no_grad():
        dummy = torch.rand(1,3,224,224).cuda()
        feature_dim = model.image_encoder(dummy).shape[-1]
        dataset_size = len(data_loader.dataset)
        datas = np.zeros([dataset_size, feature_dim])
        label_true = np.zeros(dataset_size)
        ii = 0
        print("dtype_edited")
        for x, target in data_loader:
            b = x.shape[0]
            x = Variable(x).cuda()
            image_representation = model.image_encoder(x.type(model.dtype))
            image_representation = image_representation / image_representation.norm(dim=-1, keepdim=True)
            u = image_representation.cpu()
            datas[ii * data_loader.batch_size:(ii + 1) *
                  data_loader.batch_size, :] = u.data.numpy()
            label_true[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(datas)
        label_pred = kmeans.labels_
        centroids = kmeans.cluster_centers_
        ACC = acc(label_true, label_pred, num_cluster)
        NMI = nmi(label_true, label_pred)
        ARI = ari(label_true, label_pred)
        print('image_ACC', ACC)
        print('image_NMI', NMI)
        print('image_ARI', ARI)
        return centroids, label_pred, ACC, NMI, ARI


def kmeans_with_init(model, data_loader, num_cluster, text_centroids):
    model.eval()
    with torch.no_grad():
        x = torch.rand(1,3,224,224).cuda()
        feature_dim = model.image_encoder(x).shape[-1]
        dataset_size = len(data_loader.dataset)
        datas = np.zeros([dataset_size, feature_dim])
        label_true = np.zeros(dataset_size)
        ii = 0
        text_centroids = text_centroids.cpu().numpy()
        for x, target in data_loader:
            b = x.shape[0]
            x = Variable(x).cuda()
            image_representation = model.image_encoder(x.type(model.dtype))
            # image_representation = image_representation / \
            #     image_representation.norm(dim=-1, keepdim=True)
            u = image_representation.cpu()
            datas[ii * data_loader.batch_size:(ii + 1) *
                  data_loader.batch_size, :] = u.data.numpy()
            label_true[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=num_cluster,
                        init=text_centroids, max_iter=1, n_init=1).fit(datas)
        label_pred = kmeans.labels_
        centroids = kmeans.cluster_centers_
        ACC = acc(label_true, label_pred, num_cluster)
        NMI = nmi(label_true, label_pred)
        ARI = ari(label_true, label_pred)
        print('image_ACC', ACC)
        print('image_NMI', NMI)
        print('image_ARI', NMI)
        return label_pred, ACC, NMI, ARI


def cosine_kmeans_with_init(model, data_loader, num_cluster, text_centroids):
    model.eval()
    with torch.no_grad():
        dataset_size = len(data_loader.dataset)
        label_pred = np.zeros(dataset_size)
        label_true = np.zeros(dataset_size)
        ii = 0
        text_centroids = text_centroids / \
            text_centroids.norm(dim=-1, keepdim=True)
        for x, target in data_loader:
            b = x.shape[0]
            x = Variable(x).cuda()
            image_representation = model.image_encoder(x)
            image_representation = image_representation / \
                image_representation.norm(dim=-1, keepdim=True)
            # batch * 768 @ 768 * 10 == batch * 10
            prob = image_representation @ text_centroids.T
            pred = torch.argmax(prob, dim=-1)
            pred = pred.cpu().numpy()
            label_pred[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = pred
            label_true[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = target.numpy()
            ii = ii + 1
        ACC = acc(label_true, label_pred, num_cluster)
        NMI = nmi(label_true, label_pred)
        print('image_ACC', ACC)
        print('image_NMI', NMI)
        print('image_ARI', ARI)
        
        return torch.LongTensor(label_pred), ACC, NMI, ARI



def index_filtering_by_kmeans(model, data_loader, num_cluster, sigma=1):
    model.eval()
    with torch.no_grad():
        dataset_size = len(data_loader.dataset)
        datas = np.zeros([dataset_size, 768])
        label_true = np.zeros(dataset_size)
        ii = 0
        for x, target in data_loader:
            b = x.shape[0]
            x = Variable(x).cuda()
            image_representation = model.image_encoder(x)
            image_representation = image_representation / \
                image_representation.norm(dim=-1, keepdim=True)
            u = image_representation.cpu()
            datas[ii * data_loader.batch_size:(ii + 1) *
                  data_loader.batch_size, :] = u.data.numpy()
            label_true[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(datas)

        labels = kmeans.labels_
        dists = kmeans.transform(datas)
        dists_list = [item for item in dists]
        df = pd.DataFrame([])
        df["dists"] = dists_list
        df["labels"] = labels
        num = np.unique(labels, axis=0)
        num = num.shape[0]
        one_hot_labels = np.eye(num)[labels]
        df["label_dists"] = np.sum(dists*one_hot_labels,axis=-1)
        mean_std_list = []
        label_dists_dict = {i:[] for i in range(num_cluster)}
        for i in range(len(df)):
            row = df.iloc[i]
            label_dists_dict[row["labels"]].append(row["label_dists"])
        for i in range(num_cluster):
            mean = np.mean(label_dists_dict[i])
            std = np.std(label_dists_dict[i])
            mean_std_list.append((mean,std))
        df_temp = df
        drop_list = []
        for i in range(len(df_temp)):
            row = df_temp.iloc[i]
            label = row['labels']
            label_dists = row["label_dists"]
            mean,std = mean_std_list[label]
            if label_dists > mean + sigma*std:
                drop_list.append(i)        
        df_temp = df_temp.drop(drop_list)
        filtered_index_list = df_temp.index.tolist()
        return filtered_index_list

def index_filtering_by_kmeans_with_init(model, data_loader, num_cluster, sigma=1, text_centroids=None):
    model.eval()
    with torch.no_grad():
        dataset_size = len(data_loader.dataset)
        datas = np.zeros([dataset_size, 768])
        label_true = np.zeros(dataset_size)
        ii = 0
        text_centroids = text_centroids.cpu().numpy()
        for x, target in data_loader:
            b = x.shape[0]
            x = Variable(x).cuda()
            image_representation = model.image_encoder(x)
            # image_representation = image_representation / \
            #     image_representation.norm(dim=-1, keepdim=True)
            u = image_representation.cpu()
            datas[ii * data_loader.batch_size:(ii + 1) *
                  data_loader.batch_size, :] = u.data.numpy()
            label_true[ii * data_loader.batch_size:(ii + 1)
                       * data_loader.batch_size] = target.numpy()
            ii = ii + 1
        kmeans = KMeans(n_clusters=num_cluster,
                        init=text_centroids, max_iter=1, n_init=1).fit(datas)
        labels = kmeans.labels_
        dists = kmeans.transform(datas)
        dists_list = [item for item in dists]
        df = pd.DataFrame([])
        df["dists"] = dists_list
        df["labels"] = labels
        num = np.unique(labels, axis=0)
        num = num.shape[0]
        one_hot_labels = np.eye(num)[labels]
        df["label_dists"] = np.sum(dists*one_hot_labels,axis=-1)
        mean_std_list = []
        label_dists_dict = {i:[] for i in range(num_cluster)}
        for i in range(len(df)):
            row = df.iloc[i]
            label_dists_dict[row["labels"]].append(row["label_dists"])
        for i in range(num_cluster):
            mean = np.mean(label_dists_dict[i])
            std = np.std(label_dists_dict[i])
            mean_std_list.append((mean,std))
        df_temp = df
        drop_list = []
        for i in range(len(df_temp)):
            row = df_temp.iloc[i]
            label = row['labels']
            label_dists = row["label_dists"]
            mean,std = mean_std_list[label]
            if label_dists > mean + sigma*std:
                drop_list.append(i)        
        df_temp = df_temp.drop(drop_list)
        filtered_index_list = df_temp.index.tolist()
        ACC = acc(label_true, labels, num_cluster)
        NMI = nmi(label_true, labels)
        print('data_length:',len(label_true))
        print('image_ACC', ACC)
        print('image_NMI', NMI)
        return filtered_index_list, ACC, NMI 