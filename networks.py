
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import os

#from dassl.engine import TRAINER_REGISTRY, TrainerX
#from dassl.metrics import compute_accuracy
#from dassl.utils import load_pretrained_weights, load_checkpoint
#from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from itertools import combinations

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, num_classes, n_ctx=16, num_centroids=1):
        super().__init__()
        n_cls = num_classes
        n_ctx = n_ctx
        ctx_init = ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.num_centroids = num_centroids
        # random initialization
        print("Initializing class-specific contexts")
        # kdkd
        # ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        ctx_vectors = torch.empty(
            self.num_centroids, n_cls, n_ctx, ctx_dim, dtype=dtype)
        # kdkd
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = 'age of person' + " ".join(["X"] * n_ctx)
        # kdkd
        # prompt_prefix = "a photo of a " + prompt_prefix[12:]

        # kdkd
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        prompts = [prompt_prefix + "." for i in range(num_classes)]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(
                tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # kdkd all-age
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        # kdkd all-age
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        # kdkd all-age
        # ctx = self.ctx[:,:,3:] # ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        # ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        ctx = self.ctx
        # kdkd all-age
        prefix = self.token_prefix.expand(self.num_centroids, -1, -1, -1)
        suffix = self.token_suffix.expand(self.num_centroids, -1, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (3,n_cls, 1, dim)
                ctx,     # (3,n_cls, n_ctx, dim)
                suffix,  # (3,n_cls, *, dim)
            ],
            dim=2,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, clip_model, num_classes, n_ctx=16, num_centroids=1):
        super().__init__()
        self.prompt_learner = PromptLearner(
            clip_model, num_classes, n_ctx=n_ctx, num_centroids=num_centroids)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.maximum_similarity = 0.9 * torch.ones(10).cuda()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def cos_sim_mean(self, many_centroids):
        similarity = 0
        items = [i for i in range(len(many_centroids))]
        combi_list = list(combinations(items, 2))
        for i in range(len(combi_list)):
            a, b = combi_list[i]
            similarity += self.cos(many_centroids[a], many_centroids[b])
        return similarity/len(combi_list)

    def forward(self, image):
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        for i in range(len(prompts)):
            text_feature = self.text_encoder(prompts[i], tokenized_prompts)
            text_feature = text_feature / \
                text_feature.norm(dim=-1, keepdim=True)
            logit = image_features @ text_feature.t()
            logit = logit_scale * logit
            if i == 0:
                logits = logit.expand(1, -1, -1)
                text_features = text_feature.expand(1, -1, -1)
            else:
                logit = logit.expand(1, -1, -1)
                text_feature = text_feature.expand(1, -1, -1)
                logits = torch.cat((logits, logit), dim=0)
                text_features = torch.cat((text_features, text_feature), dim=0)
        logits = logits.mean(dim=0)
        maximum_similarity = self.maximum_similarity
        if len(text_features) > 1:
            centroids_similarity = self.cos_sim_mean(text_features)
            centroids_margin_loss = torch.nn.MSELoss()(
                centroids_similarity-0.1, centroids_similarity)
        else:
            centroids_similarity = torch.ones((10)).cuda()
            centroids_margin_loss = 0
        # kdkd
        # logits = (image_features @ text_features.t()).softmax(dim=-1)

        # kdkd
        # print("logit_scale:",logit_scale)
        # print("logits_before_scale",(image_features @ text_features.t()))
        # print("logits.shape:",torch.mean(image_features @ text_features.t(),dim=0).shape)
        # check
        # cosine similarity check -1 ~ 1
        # logit_scale check
        # contrastive learning loss try

        # kdkd
        # logits = (logit_scale * image_features @
        #           text_features.t()).softmax(dim=-1)
        # kdkds

        return logits, centroids_similarity, centroids_margin_loss


def load_clip_to_cpu(backbone_name="ViT-L/14"):
    model, preprocess = clip.load(backbone_name)
    model.eval()
    model.float()
    model.cpu()
    return model, preprocess


def save_model(clip_backbone, dataset_name, num_centroids, model, optimizer, current_epoch, cluster_name=False):
    if cluster_name:
        path = os.path.join('save', "{}_".format(cluster_name)+'{}cent'.format(str(num_centroids)))
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path,"checkpoint_{}.tar".format(current_epoch))
    else:
        path = os.path.join('save',"{}_".format(clip_backbone.replace('/', '_'))+"{}_".format(dataset_name)+'{}cent'.format(str(num_centroids)))
        os.makedirs(path, exist_ok=True)
        out = os.path.join( path,"checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'epoch': current_epoch}
    torch.save(state, out)
