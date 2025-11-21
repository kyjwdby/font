from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from src import (ContentEncoder, 
                 StyleEncoder, 
                 UNet,
                 ControlNet,
                 IDSEncoder,
                 CompontStyleFusion,
                 SkeletonEncoder,
                 SVGTransformerEncoder,
                 SVGTransformerDecoder,
                 SCR,
                 VQAdapter)
# from .embedding.component.data.adapter import VQAdapter


def build_unet(args):
    unet = UNet(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=('DownBlock2D', 
                          'MCADownBlock2D',
                          'MCADownBlock2D', 
                          'DownBlock2D'),
        up_block_types=('UpBlock2D', 
                        'StyleRSIUpBlock2D',
                        'StyleRSIUpBlock2D', 
                        'UpBlock2D'),
        block_out_channels=args.unet_channels, 
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        act_fn='silu',
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=args.style_start_channel * 16,
        attention_head_dim=1,
        channel_attn=args.channel_attn,
        content_encoder_downsample_size=args.content_encoder_downsample_size,
        content_start_channel=args.content_start_channel,
        reduction=32)
    
    return unet


def build_controlnet(args, unet):
    controlnet = ControlNet.from_unet(
        unet=unet,
        conditioning_channels=3,
        conditioning_embedding_out_channels=(16, 32, 96, 256),
        conditioning_scale=args.controlnet_conditioning_scale
        )
    
    return controlnet


def build_style_encoder(args):
    style_image_encoder = StyleEncoder(
        G_ch=args.style_start_channel,
        resolution=args.style_image_size[0])
    print("Get CG-GAN Style Encoder!")
    return style_image_encoder


def build_content_encoder(args):
    content_image_encoder = ContentEncoder(
        G_ch=args.content_start_channel,
        resolution=args.content_image_size[0])
    print("Get CG-GAN Content Encoder!")
    return content_image_encoder


def build_component_encoder(args):
    component_encoder = IDSEncoder(
        max_len=args.ids_max_len,  # 35
        n_embd=args.ids_n_embed,  # 256
        input_mode=args.ids_input_mode,  # 'ch'
        ids_mode=args.ids_mode)  # 'radical'
    print("Get Component Encoder!")
    return component_encoder


def build_component_fusioner(args):
    component_fusioner = CompontStyleFusion(
        adapter=VQAdapter(args.vqgan_path),
        c_out=args.fusioner_n_out,  # 256
        l_ids=args.ids_max_len)  # 35
    print("Get Component-Style Fusion Module!")
    return component_fusioner


def build_skeleton_encoder(args):
    skeleton_encoder = SkeletonEncoder(args)
    print("Get Skeleton Transformer Encoder!")
    return skeleton_encoder


def build_svg_encoder(args):
    # set svg_transformer
    svg_encoder = SVGTransformerEncoder(  # params_num: 82568754, self test. total_params = sum(param.numel() for param in transformer_main.parameters())
        input_channels = 1,        
        input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
        num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        depth = 2, #6,                   # depth of net. The shape of the final attention mechanism will be:
                                    # depth * (cross attention -> self_per_cross_attn * self attention)
        num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim = 1024,            # latent dimension
        cross_heads = 1,             # number of heads for cross attention. paper said 1
        latent_heads = 8,            # number of heads for latent self attention, 8
        cross_dim_head = 64,         # number of dimensions per cross attention head
        latent_dim_head = 64,        # number of dimensions per latent self attention head
        num_classes = 1000,          # output number of classes
        attn_dropout = 0.,
        ff_dropout = 0.,
        opts_dict = {'hidden_size': args.hidden_size, 'max_svg_len': args.max_svg_len, 'svg_emb_len': args.svg_emb_len,
                     'loss_w_cmd': args.loss_w_cmd, 'loss_w_args': args.loss_w_args, 'loss_w_aux': args.loss_w_aux, 'loss_w_smt': args.loss_w_smt},
        weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        self_per_cross_attn = 2      # number of self attention blocks per cross attention
        )
    print("Get DeepVecFont-V2 svg Transformer!")
    return svg_encoder


def build_svg_decoder(args):
    svg_decoder = SVGTransformerDecoder(args)
    print("Get DeepVecFont-V2 svg Transformer_decoder!")
    return svg_decoder


def build_scr(args):
    scr = SCR(
        temperature=args.temperature,
        mode=args.mode,
        image_size=args.scr_image_size)
    print("Loaded SCR module for supervision successfully!")
    return scr


def build_ddpm_scheduler(args):
    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=args.beta_scheduler,
        trained_betas=None,
        variance_type="fixed_small",
        clip_sample=True)
    return ddpm_scheduler