import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from accelerate import Accelerator  ### !!!

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_controlnet,
                 build_content_encoder,
                 build_style_encoder,
                 build_component_encoder,
                 build_component_fusioner,
                 build_skeleton_encoder,
                 build_svg_encoder,
                 build_svg_decoder,
                )
# from src.model_diffuser import FontDiffuserModelDPM  #####
# from src.dpm_solver.my_pipeline_dpm_solver import FontDiffuserDPMPipeline
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style,
                   save_image_with_content_allstyle,   ### self add
                   save_image_with_content_allstyle_v1,   ### self add
                   )

# from src.modules.style_modules import StyleAttention, StyleModulator
# from src.iffont.data.adapter import VQAdapter
from src import StyleAttention, StyleModulator, VQAdapter
from src.embedding.svg.transformers import numericalize
from src.embedding.component.data import valid_characters


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False, 
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_root", type=str, default=None)  ###
    parser.add_argument("--style_image_root", type=str, default=None)  ###
    parser.add_argument("--skeleton_image_root", type=str, default=None)  ###
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--skeleton_image_path", type=str, default=None)
    parser.add_argument("--target_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    parser.add_argument("--style_font", type=int, default=None)  ###
    # parser.add_argument("--style_char", type=int, default=None)  ###
    parser.add_argument("--style_char", type=int, nargs="+", default=[])  ###
    parser.add_argument("--content_font", type=int, default=None)  ###
    parser.add_argument("--content_char", type=int, default=None)  ###
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    #####
    num_font, num_img = len(str(len(os.listdir(args.style_image_root)))), 4
    # num_font, num_img = 3,4
    # style_font = '{n:0{w}}'.format(n=int(args.style_font), w=num_font)  # use all style fonts
    content_font = '{n:0{w}}'.format(n=int(args.content_font), w=num_font)
    # style_char = '{n:0{w}}'.format(n=int(args.style_char), w=num_img)
    style_char = ['{n:0{w}}'.format(n=int(char), w=num_img) for char in args.style_char]  ### n refs
    content_char = '{n:0{w}}'.format(n=int(args.content_char), w=num_img)
    # args.style_image_path = os.path.join(args.style_image_root, 
    #                                        '{n:0{w}}'.format(n=int(style_font), w=num_font),
    #                                        '{n:0{w}}.png'.format(n=int(style_char), w=num_img))
    args.style_image_path = [[os.path.join(args.style_image_root,   ### n refs
                                           font,
                                           '{n:0{w}}.png'.format(n=int(char), w=num_img))
                                           for char in args.style_char] 
                                           for font in sorted(os.listdir(args.style_image_root))]
    args.content_image_path = os.path.join(args.content_image_root, 
                                           '{n:0{w}}'.format(n=int(content_font), w=num_font),
                                           '{n:0{w}}.png'.format(n=int(content_char), w=num_img))
    args.skeleton_image_path = os.path.join(args.skeleton_image_root, 
                                           '{n:0{w}}'.format(n=int(content_font), w=num_font),
                                           '{n:0{w}}.png'.format(n=int(content_char), w=num_img))
    args.target_image_path = [os.path.join(args.content_image_root, 
                                           '{n:0{w}}'.format(n=int(font), w=num_font),
                                           '{n:0{w}}.png'.format(n=int(content_char), w=num_img))
                                           for font in sorted(os.listdir(args.style_image_root))]
    #####

    return args


def image_process(args, content_image=None, style_image=None):
    # 初始化骨架图像为None
    skeleton_image = None
    
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            # 只有在使用controlnet时才加载骨架图像
            if args.use_controlnet:
                skeleton_image = Image.open(args.skeleton_image_path).convert('RGB')
            content_image_pil = None
        # style_image = Image.open(args.style_image_path).convert('RGB')
        # style_image = [Image.open(path).convert('RGB') for path in args.style_image_path]  ### n refs
        style_image = [[Image.open(path).convert('RGB')
                        for path in args.style_image_path[i]]
                        for i in range(len(args.style_image_path))]  ### all styles, n refs
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None
        
    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, \
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    content_image = content_inference_transforms(content_image)[None, :]
    if skeleton_image is not None:
        skeleton_image = content_inference_transforms(skeleton_image)[None, :]
    # style_image = style_inference_transforms(style_image)[None, :]
    # style_image = torch.Tensor(np.array([style_inference_transforms(image) for image in style_image]))[None, :]  ### n refs
    style_image = torch.Tensor(np.array([[style_inference_transforms(image)
                                          for image in style_image[i]]
                                          for i in range(len(style_image))]))[None, :]  ### n refs

    return content_image, style_image, content_image_pil, skeleton_image


def prepare_style_feature_cond(args, modules, datas):
    
    if args.use_component:
        component_encoder = modules['component_encoder']
        component_fusioner = modules['component_fusioner']
        x_ch = datas["x_ch"]
        c_ch = datas["c_ch"]
        c_idx = datas["c_idx"]
        ids_embed, ids_embed2 = component_encoder(x_ch)# [B, max_len, n_embd]
        sim = component_encoder.coverage(x_ch, c_ch)  # [B,n]
        # print("x_ch: ", x_ch)
        # print("c_ch: ", c_ch)
        # print("c_idx shape: ", c_idx.shape)
        # print("ids_embed2 shape: ", ids_embed2.shape)
        # print("sim shape: ", sim.shape)
        fused_component_feat, moco_cl = component_fusioner(c_idx, ids_embed2, sim)
        # print("fused_component_feat: ", fused_component_feat.shape)

    if args.use_skeleton:
        skeleton_encoder = modules['skeleton_encoder']
        style_skeletons = datas["style_skeleton"]  ###
        # skeleton_encoder = skeleton_encoder.to(style_skeletons.device)
        skeleton_feat = skeleton_encoder(style_skeletons)
        # print("skeleton_feat: ", skeleton_feat.shape)
    
    if args.use_svg:
        svg_encoder = modules['svg_encoder']
        # print("--------------------style_svg: ", datas["style_svg"].shape)
        # print("--------------------style_svg_mask: ", datas["style_svg_mask"].shape)
        # svg_feat, _ = svg_encoder(datas["style_svg"], datas["style_svg_mask"])
        svg_feat, _ = svg_encoder(None, svg=datas["style_svg"].transpose(0,1), mask=datas["style_svg_mask"])
        # print("svg_feat: ", svg_feat.shape)
    
    if (int(args.use_component) + int(args.use_skeleton) + int(args.use_svg)) > 1:
        style_attn = modules['style_attn']

    fused_style_feature = None
    if not args.use_component and not args.use_skeleton and not args.use_svg:
        pass
    
    if args.use_component and not args.use_skeleton and not args.use_svg:
        pass
    
    if args.use_skeleton and not args.use_component and not args.use_svg:
        pass
    
    if args.use_svg and not args.use_component and not args.use_skeleton:
        pass
    
    if args.use_component and args.use_skeleton and not args.use_svg:
        fused_style_feature = style_attn(fused_component_feat, skeleton_feat)
    
    if args.use_component and args.use_svg and not args.use_skeleton:
        pass
    
    if args.use_skeleton and args.use_svg and not args.use_component:
        pass
    
    if args.use_component and args.use_skeleton and args.use_svg:
        concat_feat = torch.cat((skeleton_feat, svg_feat), dim=1)
        fused_style_feature = style_attn(fused_component_feat, concat_feat)

    # print("fused_style_feature: ", fused_style_feature.shape)
    return fused_style_feature




def prepare_data(args, style_font, style_chars, font_idx):
    datas = {}
    if args.use_component:
        # corpus = valid_characters.valid_ch
        corpus_test = valid_characters.test_ch
        x_ch: str = corpus_test[int(args.content_char)]        
        # c_ch = tuple(corpus_test[int(args.style_char)])
        c_ch = tuple([corpus_test[int(char)] for char in args.style_char])  ### n refs
        
        adapter = VQAdapter(args.vqgan_path)
        adapter.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # style_image_ori = Image.open(args.style_image_path).convert('RGB')
        # style_image_ori = [Image.open(path).convert('RGB') for path in args.style_image_path]  ### n refs
        style_image_ori = [Image.open(path).convert('RGB')
                            for path in args.style_image_path[font_idx]]  ### font_idx style!, n refs
        noresize_transform = transforms.Compose(
                [transforms.Resize((128, 128), 
                                interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])])
        # style_image_ori = noresize_transform(style_image_ori)[None, :]
        # c_idx = adapter.encode(style_image_ori)
        # style_image_ori = np.array([noresize_transform(image) for image in style_image_ori])  ### n refs
        # c_idx = np.array([adapter.encode(img).cpu() for img in style_image_ori])  ### n refs
        style_image_ori = np.array([noresize_transform(image)
                                    for image in style_image_ori])  ### font_idx style!, n refs
        c_idx = np.array([adapter.encode(img).cpu()
                        for img in style_image_ori])  ### font_idx style!, n refs
        c_idx = torch.as_tensor(c_idx, dtype=torch.long).to(args.device)
        # c_idx = c_idx.unsqueeze(0).unsqueeze(0)  # old 1 ref
        c_idx = c_idx.unsqueeze(0)  ### n refs
        # print("--------------- c_idx shape: ", c_idx.shape)
        ###################################################
        uncond_x_ch = '一'
        uncond_c_ch = tuple('一' for i in range(len(c_ch)))
        uncond_c_idx = torch.zeros_like(c_idx)
        datas["x_ch"] = [uncond_x_ch, x_ch]
        datas["c_ch"] = [uncond_c_ch, c_ch]
        datas["c_idx"] = torch.cat([uncond_c_idx, c_idx], dim=0)

    if args.use_skeleton:
        style_skeleton_fonti = []
        # style_fonts = sorted(os.listdir(args.style_image_root))
        for style_char_i in style_chars:
            style_skeleton_i = np.load(os.path.join(args.skeleton_root, style_font, '{}_{}_bone_sdf.npy'.format(style_font, style_char_i)))
            if style_skeleton_i.shape[0] < args.max_skeleton_len:
                style_skeleton_i = np.vstack([style_skeleton_i, np.zeros((args.max_skeleton_len-style_skeleton_i.shape[0], style_skeleton_i.shape[1]))])
            style_skeleton_fonti.append(style_skeleton_i)
        style_skeletons = torch.FloatTensor(np.array(style_skeleton_fonti)).unsqueeze(0).to(args.device)   # [1, refs, N,12]

        # target_skeleton_i = np.load(os.path.join(args.skeleton_root, style_fonts[i], '{}_{}_bone_sdf.npy'.format(style_fonts[i], content_char)))
        # if target_skeleton_i.shape[0] < args.max_skeleton_len:
        #     target_skeleton_i = np.vstack([target_skeleton_i, np.zeros((args.max_skeleton_len-target_skeleton_i.shape[0], target_skeleton_i.shape[1]))])
        # target_skeletons = torch.FloatTensor(np.array(target_skeleton_i)).unsqueeze(0).to(args.device)   # [1, N,12]
        
        ##################### exclude sdf !!!!!!!!!!!!!!!
        style_skeletons = torch.cat([style_skeletons[:,:,:,:2], style_skeletons[:,:,:,-1:]], dim=-1)
        # content_skeletons = torch.cat([content_skeletons[:,:,:2], content_skeletons[:,:,-1:]], dim=-1)
        # target_skeletons = torch.cat([target_skeletons[:,:,:2], target_skeletons[:,:,-1:]], dim=-1)
        #####################
        uncond_style_skeletons = torch.zeros_like(style_skeletons)
        datas["style_skeleton"] = torch.cat([uncond_style_skeletons, style_skeletons], dim=0) 

    if args.use_svg:
        # 1 style svg now, 3 refs to do later
        if os.path.exists(os.path.join(args.svg_root, style_font, '{}_{}_seq_len.npy'.format(style_font, style_chars[0]))):
            style_svg_len = torch.LongTensor(np.load(os.path.join(args.svg_root, style_font,  # [1]
                        '{}_{}_seq_len.npy'.format(style_font, style_chars[0]))))
            style_svg = torch.FloatTensor(np.load(os.path.join(args.svg_root, style_font, # [261, 12]
                        '{}_{}_sequence_relaxed.npy'.format(style_font, style_chars[0]))))
            if style_svg.shape[0] < args.max_svg_len:
                style_svg = torch.cat([style_svg, torch.zeros((args.max_svg_len-style_svg.shape[0], style_svg.shape[1]))], dim=0)
        else:
            style_svg_len = torch.LongTensor([0])
            style_svg = torch.FloatTensor(np.zeros((args.max_svg_len, args.dim_svg)))
        # content_svg_len = torch.LongTensor(np.load(os.path.join(args.svg_root, content_font,  # [1]
        #             '{}_{}_seq_len.npy'.format(content_font, content_char))))
        # content_svg = torch.FloatTensor(np.load(os.path.join(args.svg_root, content_font, # [261, 12]
        #             '{}_{}_sequence_relaxed.npy'.format(content_font, content_char))))
        # target_svg_len = torch.LongTensor(np.load(os.path.join(args.svg_root, style_font,  # [1]
        #             '{}_{}_seq_len.npy'.format(style_font, content_char))))
        # target_svg = torch.FloatTensor(np.load(os.path.join(args.svg_root, style_font, # [261, 12]
        #             '{}_{}_sequence_relaxed.npy'.format(style_font, content_char))))
        # style_thickthin = torch.FloatTensor(np.load(os.path.join(args.feat_root, style_font, # [2]
        #             '{}_{}_thickthin.npy'.format(style_font, style_char))))
        
        arg_quant = numericalize(style_svg[:, 4:])
        cmd_cls = torch.argmax(style_svg[:, :4], dim=-1).unsqueeze(-1)
        style_svg = torch.cat([cmd_cls, arg_quant], dim=-1) # 1 + 8 = 9 dimension, 12->9
        # arg_quant = numericalize(content_svg[:, 4:])
        # cmd_cls = torch.argmax(content_svg[:, :4], dim=-1).unsqueeze(-1)
        # content_svg = torch.cat([cmd_cls, arg_quant], dim=-1)
        # # mask
        style_svg_mask = torch.zeros(1, args.svg_emb_len)
        if style_svg_len[0] > 0:
            denom = args.max_svg_len / args.svg_emb_len *1.0
            style_svg_mask[:, :int(np.ceil( (style_svg_len[0]+1)/denom ))] = 1
        # content_svg_mask = torch.zeros(1, args.svg_emb_len) # [1,9] value = 1 means pos to be masked
        # content_svg_mask[:, :int(np.ceil( (content_svg_len[0]+1)/denom ))] = 1

        style_svg = style_svg.unsqueeze(0).to(args.device)
        style_svg_mask = style_svg_mask.unsqueeze(0).to(args.device)
        uncond_style_svg = torch.zeros_like(style_svg)
        uncond_style_svg_mask = torch.zeros_like(style_svg_mask)
        datas["style_svg"] = torch.cat([uncond_style_svg, style_svg], dim=0)
        datas["style_svg_mask"] = torch.cat([uncond_style_svg_mask, style_svg_mask], dim=0)

    return datas



def load_modules_state_old(modules, ckpt_path, device):
    """
    从 accelerator.save_state() 保存的权重文件中加载各模块参数到 modules 字典。
    参数：
        modules: nn.ModuleDict 或 普通字典
        ckpt_path: 保存的 pytorch_model.bin 路径
        device: torch.device，如 torch.device("cuda:0")
    """
    # 1. 读取完整模型字典
    print(f"Loading checkpoint from {ckpt_path} ...")
    full_state = torch.load(ckpt_path, map_location=device)

    for k, v in full_state.items():  # !!!!!
        print(f"Key: {k}, Value type: {type(v)}")

    # 如果保存的是 accelerate 格式，模型参数一般在 "model" 或 "model_0" 等 key 中
    if any(k.startswith("model_") for k in full_state.keys()):
        # 加速器的多卡保存形式
        merged_state = {}
        for k, v in full_state.items():
            if k.startswith("model_"):
                merged_state.update(v)
        full_state = merged_state

    # 2. 工具函数：从完整 state_dict 提取子模块参数
    def extract_subdict(state_dict, prefix):
        prefix_dot = prefix + "."
        return {k[len(prefix_dot):]: v for k, v in state_dict.items() if k.startswith(prefix_dot)}

    # 3. 遍历 modules 字典并加载各模块权重
    for name, module in modules.items():
        sub_state = extract_subdict(full_state, f"modules.{name}")
        if len(sub_state) == 0:
            print(f"[WARN] No weights found for module '{name}' — skipping.")
            continue
        missing, unexpected = module.load_state_dict(sub_state, strict=False)
        print(f"[OK] Loaded '{name}' — missing: {len(missing)}, unexpected: {len(unexpected)}")

    print("✅ All modules loaded successfully.")





def load_modules_state(modules, full_state):
    """
    将 full_state 中的子模块权重加载回 modules (nn.ModuleDict)
    """
    for name, module in modules.items():
        # 1. 找出属于该子模块的参数
        sub_state_dict = {k.replace(f"{name}.", ""): v 
                          for k, v in full_state.items() if k.startswith(f"{name}.")}

        if len(sub_state_dict) == 0:
            print(f"[Warning] No weights found for module: {name}")
            continue

        # 2. 加载
        missing, unexpected = module.load_state_dict(sub_state_dict, strict=False)

        print(f"[Loaded] {name}: {len(sub_state_dict)} tensors loaded.")
        if missing:
            print(f"  Missing keys: {missing}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")




def load_fontdiffuer_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    if args.use_controlnet:
        controlnet = build_controlnet(args=args, unet=unet)
    else:
        controlnet = None
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    ##################################### multi-modal encoders
    # modules = {}
    modules = nn.ModuleDict()  #####
    # params = []
    if args.use_component:
        component_encoder = build_component_encoder(args)
        component_encoder.set_device(args.device)
        component_fusioner = build_component_fusioner(args)
        component_fusioner.set_device(args.device)
        modules['component_encoder'] = component_encoder
        modules['component_fusioner'] = component_fusioner
    else:
        component_encoder = None
        component_fusioner = None
    if args.use_skeleton:
        skeleton_encoder = build_skeleton_encoder(args)
        modules['skeleton_encoder'] = skeleton_encoder
    else:
        skeleton_encoder = None
    if args.use_svg:
        svg_encoder = build_svg_encoder(args=args)
        svg_decoder = build_svg_decoder(args=args)
        modules['svg_encoder'] = svg_encoder
        modules['svg_decoder'] = svg_decoder
    else:
        svg_encoder = None
        svg_decoder = None
    if int(args.use_component) + int(args.use_skeleton) + int(args.use_svg) > 1:
        style_attn = StyleAttention()
        modules['style_attn'] = style_attn
    else:
        style_attn = None
    #####################################
    if args.use_controlnet:
        style_modulator = StyleModulator()  # default fixed param, to be inputed
    else:
        style_modulator = None
    
    # model = FontDiffuserModelDPM(
    #     unet=unet,
    #     style_encoder=style_encoder,
    #     content_encoder=content_encoder)


    # 从训练 checkpoint 加载权重
    # ------------------------------
    device = torch.device(f"{args.device}" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(os.path.join(args.ckpt_dir, "pytorch_model.bin"), map_location=device)

    # 一些 accelerator 会封装 key，如 "module.unet.xxx"
    def extract_subdict(state_dict, prefix):
        return {k[len(prefix)+1:]: v for k, v in state_dict.items() if k.startswith(prefix)}

    ### 提取model_diffuser每个子模块的权重
    unet_state = extract_subdict(model_state, "unet")
    controlnet_state = extract_subdict(model_state, "controlnet") if args.use_controlnet else None
    style_enc_state = extract_subdict(model_state, "style_encoder")
    content_enc_state = extract_subdict(model_state, "content_encoder")
    style_mod_state = extract_subdict(model_state, "style_modulator") if style_modulator else None

    # 分别加载
    missing, unexpected = unet.load_state_dict(unet_state, strict=False)
    print(f"[unet] missing: {missing}, unexpected: {unexpected}")
    
    if controlnet and controlnet_state:
        controlnet.load_state_dict(controlnet_state, strict=False)
    style_encoder.load_state_dict(style_enc_state, strict=False)
    content_encoder.load_state_dict(content_enc_state, strict=False)
    if style_modulator and style_mod_state:
        style_modulator.load_state_dict(style_mod_state, strict=False)
        
    model_diffuser = FontDiffuserModelDPM(
        unet=unet,
        controlnet=controlnet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        style_modulator=style_modulator,
        )
    model_diffuser.to(args.device)
    print("Loaded model weights into FontDiffuserModelDPM successfully.")


    ### 提取modules
    module_state = torch.load(os.path.join(args.ckpt_dir, "pytorch_model_1.bin"), map_location=device)
    # load_modules_state(modules, full_state)
    
    component_encoder_state = extract_subdict(module_state, "component_encoder") if args.use_component else None
    component_fusioner_state = extract_subdict(module_state, "component_fusioner") if args.use_component else None
    skeleton_encoder_state = extract_subdict(module_state, "skeleton_encoder") if args.use_skeleton else None
    svg_encoder_state = extract_subdict(module_state, "svg_encoder") if args.use_svg else None
    style_attn_state = extract_subdict(module_state, "style_attn") if (int(args.use_component) + int(args.use_skeleton) + int(args.use_svg)) > 1 else None
    # 分别加载
    if args.use_component and component_encoder_state:
        modules['component_encoder'].load_state_dict(component_encoder_state, strict=False)
        modules['component_fusioner'].load_state_dict(component_fusioner_state, strict=False)
    if args.use_skeleton and skeleton_encoder_state:
        modules['skeleton_encoder'].load_state_dict(skeleton_encoder_state, strict=False)
    if args.use_svg and svg_encoder_state:
        modules['svg_encoder'].load_state_dict(svg_encoder_state, strict=False)
    if style_attn_state:
        modules['style_attn'].load_state_dict(style_attn_state, strict=False)
    modules.to(args.device)
    print("Loaded the modules state_dict successfully!")
    
    ####### model_bone in sampling
    # model_bone = seq_branch(args)
    # model_bone.to(args.device)

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model_diffuser,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe, modules


def sampling(args, pipe, modules, content_image=None, style_image=None):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)
    
    content_image, style_image, content_image_pil, skeleton_image = image_process(args=args, 
                                                                  content_image=content_image, 
                                                                  style_image=style_image)
    ###################################################
    # style_font, style_char = '19', '001'
    # content_font, content_char = '08', '000'
    # num_font, num_char = 2,3
    num_font, num_char = len(str(len(os.listdir(args.style_image_root)))),4
    # style_font = '{n:0{w}}'.format(n=int(args.style_font), w=num_font)
    content_font = '{n:0{w}}'.format(n=int(args.content_font), w=num_font)
    # style_char = '{n:0{w}}'.format(n=int(args.style_char), w=num_char)
    style_char = ['{n:0{w}}'.format(n=int(char), w=num_char) for char in args.style_char]  ### n refs
    content_char = '{n:0{w}}'.format(n=int(args.content_char), w=num_char)

    # style_seq_len = torch.LongTensor(np.load(os.path.join(args.seq_root, style_font,  # [1]
    #             '{}_{}_seq_len.npy'.format(style_font, style_char)))).to(args.device)
    # style_seq = torch.FloatTensor(np.load(os.path.join(args.seq_root, style_font, # [261, 12]
    #             '{}_{}_sequence_relaxed.npy'.format(style_font, style_char)))).to(args.device)
    # content_seq_len = torch.LongTensor(np.load(os.path.join(args.seq_root, content_font,  # [1]
    #             '{}_{}_seq_len.npy'.format(content_font, content_char)))).to(args.device)
    # content_seq = torch.FloatTensor(np.load(os.path.join(args.seq_root, content_font, # [261, 12]
    #             '{}_{}_sequence_relaxed.npy'.format(content_font, content_char)))).to(args.device)
    # style_thickthin = torch.FloatTensor(np.load(os.path.join(args.my_feat_root, style_font, # [2]
    #             '{}_{}_thickthin.npy'.format(style_font, style_char)))).to(args.device)

    # from src.modules.seq.transformers import numericalize
    # arg_quant = numericalize(style_seq[:, 4:])
    # cmd_cls = torch.argmax(style_seq[:, :4], dim=-1).unsqueeze(-1)
    # style_seq = torch.cat([cmd_cls, arg_quant], dim=-1) # 1 + 8 = 9 dimension, 12->9
    # arg_quant = numericalize(content_seq[:, 4:])
    # cmd_cls = torch.argmax(content_seq[:, :4], dim=-1).unsqueeze(-1)
    # content_seq = torch.cat([cmd_cls, arg_quant], dim=-1)
    # # mask
    # style_seq_mask = torch.zeros(1, args.seq_emb_len).to(args.device)
    # content_seq_mask = torch.zeros(1, args.seq_emb_len).to(args.device) # [1,9] value = 1 means pos to be masked
    # denom = args.max_seq_len / args.seq_emb_len *1.0
    # style_seq_mask[:, :int(torch.ceil( (style_seq_len[0]+1)/denom ))] = 1
    # content_seq_mask[:, :int(torch.ceil( (content_seq_len[0]+1)/denom ))] = 1

    # style_seq, content_seq, style_seq_mask, content_seq_mask = \
    #     style_seq.unsqueeze(0), content_seq.unsqueeze(0), style_seq_mask.unsqueeze(0), content_seq_mask.unsqueeze(0)
    # seq_data = {'style_seq':style_seq, 'content_seq':content_seq, 'style_seq_mask':style_seq_mask, 'content_seq_mask':content_seq_mask}
    # style_thickthin = style_thickthin.unsqueeze(0).unsqueeze(1)  ##### [B,1,2]
    # feat_data = {'style_thickthin':style_thickthin}
    
    ### content bone
    # content_bones = torch.FloatTensor(np.load(os.path.join(args.bone_root, content_font, # [N,12]
    #             '{}_{}_bone_sdf.npy'.format(content_font, content_char))))
    # if content_bones.shape[0] < args.max_bone_len:
    #     content_bones = torch.cat([content_bones, torch.zeros((args.max_bone_len-content_bones.shape[0], content_bones.shape[1]))], dim=0)
    # # content_bones = content_bones.unsqueeze(0).repeat(target_bones.shape[0], 1, 1).to(args.device)  #####
    # content_bones = content_bones.unsqueeze(0).to(args.device)  #####
    
    
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        # 只有在使用controlnet且skeleton_image不为None时才进行设备转换
        if args.use_controlnet and skeleton_image is not None:
            skeleton_image = skeleton_image.to(args.device)
        ##################################
        # # model_bone = seq_branch(args)
        # model_bone = style_seq_encoder(args)
        # model_bone.load_state_dict(torch.load(f"{args.ckpt_dir}/model_bone.pth"))
        # model_bone.to(args.device)
        # # output_bone = model_bone(content_bones, style_bones, target_bones, device=0)
        ##################################
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        ##################################
        # print("style path: ", np.array(args.style_image_path)[:,0])
        # print("target path: ", args.target_image_path)
        # print("style_image shape: ", style_image.shape)  # [1, 34, 3, 3, 96, 96]
        images_all = []
        for i in range(style_image.shape[1]):
            
            style_fonts = sorted(os.listdir(args.style_image_root))
            style_font = style_fonts[i]

            datas = prepare_data(args, style_font, style_char, i)

            style_feature_cond = prepare_style_feature_cond(args, modules, datas)
            
            ##################################
            # cross modal loss test
            # # target_image = Image.open(args.target_image_path[i]).convert('L')
            # target_image_path_wrong = os.path.join(args.content_image_root, 
            #             '{n:0{w}}'.format(n=i, w=num_font),
            #             '{n:0{w}}.png'.format(n=int(args.content_char+1), w=4))#num_img))
            # target_image = Image.open(target_image_path_wrong).convert('L')
            # target_transforms = transforms.Compose(
            #     [transforms.Resize(args.resolution, \
            #                     interpolation=transforms.InterpolationMode.BILINEAR),
            #     transforms.ToTensor()])
            # target_image = target_transforms(target_image).to(args.device)
            # from src.loss import cross_modal_loss_normalized
            # (pi, mu, sigma, rho, label) = output_bone['fake']
            # loss, pred_heatmap = cross_modal_loss_normalized(
            #     pi, mu, sigma, rho, target_image, H=args.resolution, W=args.resolution, sigma_heatmap=0.03
            # )
            # print("cross modal loss: ", loss)
            ##################################
            # 始终传递skeleton_images参数，但在不使用controlnet时设为None
            images = pipe.generate(
                content_images=content_image,
                style_images=style_image[:,i],
                skeleton_images=None if not args.use_controlnet else skeleton_image,
                style_feature_cond=style_feature_cond,
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size,
                algorithm_type=args.algorithm_type,
                skip_type=args.skip_type,
                method=args.method,
                correcting_x0_fn=args.correcting_x0_fn)
            images_all.append(images[0])
        end = time.time()
        ##################################

        if args.save_image:
            print(f"Saving the image ......")
            # save_single_image(save_dir=args.save_image_dir, image=images[0])
            # if args.character_input:
            #     save_image_with_content_style(save_dir=args.save_image_dir,
            #                                 image=images[0],
            #                                 content_image_pil=content_image_pil,
            #                                 content_image_path=None,
            #                                 # style_image_path=args.style_image_path,
            #                                 style_image_path=args.style_image_path[0],  ### n refs
            #                                 target_image_path=args.target_image_path,  ###
            #                                 resolution=args.resolution)
            # else:
            #     save_image_with_content_style(save_dir=args.save_image_dir,
            #                                 image=images[0],
            #                                 content_image_pil=None,
            #                                 content_image_path=args.content_image_path,
            #                                 # style_image_path=args.style_image_path,
            #                                 style_image_path=args.style_image_path[0],  ### n refs
            #                                 target_image_path=args.target_image_path,  ###
            #                                 resolution=args.resolution)

            ##################################
            # save_image_with_content_allstyle(save_dir=args.save_image_dir,
            save_image_with_content_allstyle_v1(save_dir=args.save_image_dir,
                                        image=images_all,
                                        content_image_pil=None,
                                        content_image_path=args.content_image_path,
                                        # style_image_path=args.style_image_path,
                                        style_image_path=np.array(args.style_image_path)[:,0],  ### n refs
                                        target_image_path=args.target_image_path,  ###
                                        resolution=args.resolution)
            ##################################
            print(f"Finish the sampling process, costing time {end - start}s")
        # return images[0]
        return images_all


def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny", 
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    # load controlnet model and pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    controlnet = ControlNetModel.from_pretrained(config_path, 
                                                 torch_dtype=torch.float16,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path, 
                                                             controlnet=controlnet, 
                                                             torch_dtype=torch.float16,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe


def controlnet(text_prompt, 
               pil_image,
               pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(text_prompt, 
                 num_inference_steps=50, 
                 generator=generator, 
                 image=canny_image,
                 output_type='pil').images[0]
    return image


def load_instructpix2pix_pipeline(args,
                                  ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, 
                                                                  torch_dtype=torch.float16)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=text_prompt, image=image, generator=generator, 
                 num_inference_steps=20, image_guidance_scale=1.1).images[0]

    return image


if __name__=="__main__":
    args = arg_parse()
    
    # load fontdiffuser pipeline
    pipe, modules = load_fontdiffuer_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe, modules=modules)
