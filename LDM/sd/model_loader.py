from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion

import model_converter
from utils import *
def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # encoder = VAE_Encoder().to(device)
    # encoder.load_state_dict(state_dict['encoder'], strict=True)

    # decoder = VAE_Decoder().to(device)
    # decoder.load_state_dict(state_dict['decoder'], strict=True)

    # diffusion = Diffusion().to(device)
    # diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    # encoder_params = get_parameter_number(encoder)
    
    # decoder_params = get_parameter_number(decoder)
    # diffusion_params = get_parameter_number(diffusion)
    # clip_params = get_parameter_number(clip)
    # print("encoder params:",encoder_params['Total'])
    # print("decoder params:",decoder_params['Total'])
    # print("diffusion params:",diffusion_params['Total'])
    # print("clip params:",clip_params['Total'])
    
    return {
        'clip': clip,
        # 'encoder': encoder,
        # 'decoder': decoder,
        # 'diffusion': diffusion,
    }