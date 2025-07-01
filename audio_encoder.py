from transformers import ASTModel 
import os
import numpy as np
import torch
from opera.src.model.models_cola import Cola
from opera.src.model.models_mae import mae_vit_small

def initialize_pretrained_model(pretrain, opera_checkpoint_path):
    if pretrain == "operaCT":
        model = Cola(encoder="htsat")
    elif pretrain == "operaCE":
        model = Cola(encoder="efficientnet")
    elif pretrain == "operaGT":
        model = mae_vit_small(norm_pix_loss=False,
                              in_chans=1, audio_exp=True,
                              img_size=(256,64),
                              alpha=0.0, mode=0, use_custom_patch=False,
                              split_pos=False, pos_trainable=False, use_nce=False,
                              decoder_mode=1,
                              mask_2d=False, mask_t_prob=0.7, mask_f_prob=0.3,
                              no_shift=False).float()
    elif pretrain == "ast":
        model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593", trust_remote_code=True)
    else:
        raise NotImplementedError(f"Model not exist: {pretrain}, please check the parameter.")


    checkpoint = torch.load(opera_checkpoint_path, map_location='cuda')
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model
