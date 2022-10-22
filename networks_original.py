
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

#from dassl.engine import TRAINER_REGISTRY, TrainerX
#from dassl.metrics import compute_accuracy
#from dassl.utils import load_pretrained_weights, load_checkpoint
#from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


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
    def __init__(self, clip_model, num_classes, n_ctx=16):
        super().__init__()
        n_cls = num_classes
        n_ctx = n_ctx
        ctx_init = ''
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        print("Initializing class-specific contexts")
        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        ##kdkd
        # prompt_prefix = "a photo of a " + prompt_prefix[12:]
        ##kdkd
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
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, clip_model, num_classes, n_ctx=16):
        super().__init__()
        self.prompt_learner = PromptLearner(
            clip_model, num_classes, n_ctx=n_ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        ##kdkd
        # logits = (image_features @ text_features.t()).softmax(dim=-1)
        logits = image_features @ text_features.t()
        logits = logit_scale * logits
        
        
        ##kdkd
        # print("logit_scale:",logit_scale)
        # print("logits_before_scale",(image_features @ text_features.t()))
        # print("logits.shape:",torch.mean(image_features @ text_features.t(),dim=0).shape)
        ## check
        ## cosine similarity check -1 ~ 1 
        ## logit_scale check
        ## contrastive learning loss try
        
        
        # kdkd
        # logits = (logit_scale * image_features @
        #           text_features.t()).softmax(dim=-1)
        # kdkds

        return logits


def load_clip_to_cpu(backbone_name="ViT-L/14"):
    model, preprocess = clip.load(backbone_name)
    model.eval()
    model.float()
    model.cpu()
    return model, preprocess
