import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (#FontDiffuserDPMPipeline,
                 #FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_controlnet,
                 build_content_encoder,
                 build_style_encoder,
                #  build_seq_encoder,   #####
                #  build_seq_decoder,   #####
                #  build_thickthin_encoder,   #####
                )
from src.model_diffuser import FontDiffuserModelDPM  #####
from src.dpm_solver.my_pipeline_dpm_solver import FontDiffuserDPMPipeline
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style,
                   save_image_with_content_allstyle,   ### self add
                   save_image_with_content_allstyle_v1,   ### self add
                   save_generated_image,   ### self add
                   )

from src.model_diffuser import Bone_attn, StyleModulator  ###
from src.model_bone import seq_branch, style_seq_encoder  #####
from src.modules.compont_encoder import IDSEncoder
from src.modules.compont_style_fusion import CompontStyleFusion
from src.iffont.data.adapter import VQAdapter

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
    parser.add_argument("--style_font", type=int, nargs="+", default=[])  ###
    # parser.add_argument("--style_char", type=int, default=None)  ###
    parser.add_argument("--style_char", type=int, nargs="+", default=[])  ###
    parser.add_argument("--content_font", type=int, default=None)  ###
    parser.add_argument("--content_char", type=int, nargs="+", default=[])  ###

    parser.add_argument("--style_font_str", type=str, nargs="+", default=[])  ###
    parser.add_argument("--style_char_str", type=str, nargs="+", default=[])  ###
    parser.add_argument("--content_font_str", type=str, default=None)  ###
    parser.add_argument("--content_char_str", type=str, nargs="+", default=[])  ###
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    #####
    num_font, num_img = len(str(len(os.listdir(args.style_image_root)))), 4
    # num_font, num_img = 3,4
    # style_font = '{n:0{w}}'.format(n=int(args.style_font), w=num_font)
    args.style_font_str = ['{n:0{w}}'.format(n=int(font), w=num_font) for font in args.style_font]  # multiple style font
    args.content_font_str = '{n:0{w}}'.format(n=int(args.content_font), w=num_font)
    # style_char = '{n:0{w}}'.format(n=int(args.style_char), w=num_img)
    args.style_char_str = ['{n:0{w}}'.format(n=int(char), w=num_img) for char in args.style_char]  ### n refs
    # content_char = '{n:0{w}}'.format(n=int(args.content_char), w=num_img)
    args.content_char_str = ['{n:0{w}}'.format(n=int(char), w=num_img) for char in args.content_char]  ### multiple content char
    # args.style_image_path = os.path.join(args.style_image_root, 
    #                                        '{n:0{w}}'.format(n=int(style_font), w=num_font),
    #                                        '{n:0{w}}.png'.format(n=int(style_char), w=num_img))
    args.style_image_path = [[os.path.join(args.style_image_root,   ### n refs
                                           font,
                                           char+'.png')
                                           for char in args.style_char_str] 
                                           for font in args.style_font_str]
                                        #    for font in sorted(os.listdir(args.style_image_root))]
    args.content_image_path = [os.path.join(args.content_image_root, 
                                           args.content_font_str,
                                           char+'.png')
                                           for char in args.content_char_str]  ### multiple content char
    args.skeleton_image_path = [os.path.join(args.skeleton_image_root, 
                                           args.content_font_str,
                                           char+'.png')
                                           for char in args.content_char_str]
    args.target_image_path = [[os.path.join(args.content_image_root, 
                                           font,
                                           char+'.png')
                                           for char in args.content_char_str]
                                           for font in args.style_font_str]  ### multiple target path
    #####

    return args


def image_process(args, content_image=None, style_image=None):
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            # assert args.content_character is not None, "The content_character should not be None."
            # if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
            #     return None, None
            # font = load_ttf(ttf_path=args.ttf_path)
            # content_image = ttf2im(font=font, char=args.content_character)
            # content_image_pil = content_image.copy()
            raise ValueError("In non-demo mode, the character_input is not supported now.")  ## self
        else:
            # content_image = Image.open(args.content_image_path).convert('RGB')
            content_image = [Image.open(path).convert('RGB')
                             for path in args.content_image_path]  ### multiple content char
            # skeleton_image = Image.open(args.skeleton_image_path).convert('RGB')
            skeleton_image = [Image.open(path).convert('RGB')
                              for path in args.skeleton_image_path]  ### multiple content char
            content_image_pil = None
        # style_image = Image.open(args.style_image_path).convert('RGB')
        # style_image = [Image.open(path).convert('RGB') for path in args.style_image_path]  ### n refs
        style_image = [[Image.open(path).convert('RGB')
                        for path in args.style_image_path[i]]
                        for i in range(len(args.style_image_path))]  ### multiple styles, n refs
    else:
        raise ValueError("Demo mode is not supported now.")  # self
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
    # content_image = content_inference_transforms(content_image)[None, :]
    content_image = torch.stack([content_inference_transforms(image)
                                for image in content_image])[None, :]
    # skeleton_image = content_inference_transforms(skeleton_image)[None, :]
    skeleton_image = torch.stack([content_inference_transforms(image)
                                for image in skeleton_image])[None, :]
    # style_image = style_inference_transforms(style_image)[None, :]
    # style_image = torch.Tensor(np.array([style_inference_transforms(image) for image in style_image]))[None, :]  ### n refs
    # style_image = torch.Tensor(np.array([[style_inference_transforms(image)
    #                                       for image in style_image[i]]
    #                                       for i in range(len(style_image))]))[None, :]  ### n refs
    style_image = torch.stack([torch.stack([style_inference_transforms(image)
                                for image in ref_list])
                                for ref_list in style_image])[None, :]  ### n refs

    return content_image, style_image, content_image_pil, skeleton_image

def load_fontdiffuer_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    controlnet = build_controlnet(args=args, unet=unet)
    controlnet.load_state_dict(torch.load(f"{args.ckpt_dir}/controlnet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    # seq_encoder = build_seq_encoder(args=args)
    # seq_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/seq_encoder.pth"))  #####
    # thickthin_encoder = build_thickthin_encoder(args=args)
    # thickthin_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/thickthin_encoder.pth"))  #####
    bone_attn = Bone_attn()  #####
    bone_attn.load_state_dict(torch.load(f"{args.ckpt_dir}/bone_attn.pth"))
    style_modulator = StyleModulator()  #####
    style_modulator.load_state_dict(torch.load(f"{args.ckpt_dir}/style_modulator.pth"))
    ids_encoder = IDSEncoder(max_len=35, n_embd=256, input_mode='ch', ids_mode='radical')  #####
    ids_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/ids_encoder.pth"))
    ids_encoder.set_device(args.device)
    style_fusioner = CompontStyleFusion(adapter=VQAdapter(args.vqgan_path), c_out=256, l_ids=35)
    style_fusioner.load_state_dict(torch.load(f"{args.ckpt_dir}/style_fusioner.pth"))
    style_fusioner.set_device(args.device)
    # model = FontDiffuserModelDPM(
    #     unet=unet,
    #     style_encoder=style_encoder,
    #     content_encoder=content_encoder)
    model_diffuser = FontDiffuserModelDPM(
        unet=unet,
        controlnet=controlnet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        # seq_encoder=seq_encoder,  #####
        # thickthin_encoder=thickthin_encoder,  #####
        bone_attn=bone_attn,  #####
        style_modulator=style_modulator,  #####
        ids_encoder=ids_encoder,  #####
        style_fusioner=style_fusioner,  #####
        )
    model_diffuser.to(args.device)
    print("Loaded the model state_dict successfully!")
    
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

    return pipe


def sampling(args, pipe, content_image=None, style_image=None):
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
    # # style_font, style_char = '19', '001'
    # # content_font, content_char = '08', '000'
    # # num_font, num_char = 2,3
    # num_font, num_char = len(str(len(os.listdir(args.style_image_root)))),4
    # # style_font = '{n:0{w}}'.format(n=int(args.style_font), w=num_font)
    # content_font = '{n:0{w}}'.format(n=int(args.content_font), w=num_font)
    # # style_char = '{n:0{w}}'.format(n=int(args.style_char), w=num_char)
    # style_char = ['{n:0{w}}'.format(n=int(char), w=num_char) for char in args.style_char]  ### n refs
    # content_char = '{n:0{w}}'.format(n=int(args.content_char), w=num_char)
    style_font = args.style_font_str  ### multiple style font
    content_font = args.content_font_str
    style_char = args.style_char_str  ### n refs
    content_char = args.content_char_str  ### multiple content char


    ### 多风格、多内容循环  self new
    images_all = []
    model_bone = style_seq_encoder(args)
    model_bone.load_state_dict(torch.load(f"{args.ckpt_dir}/model_bone.pth"))
    model_bone.to(args.device)
    start = time.time()
    for i, font in enumerate(args.style_font_str):
        style_bones_allchars = []
        for s_char in args.style_char_str:
            style_bone_i = np.load(os.path.join(args.bone_root, font, f"{font}_{s_char}_bone_sdf.npy"))
            if style_bone_i.shape[0] < args.max_bone_len:
                style_bone_i = np.vstack([style_bone_i, np.zeros((args.max_bone_len - style_bone_i.shape[0], style_bone_i.shape[1]))])
            style_bones_allchars.append(style_bone_i)
        style_bones = torch.FloatTensor(np.array(style_bones_allchars)).unsqueeze(0).to(args.device)
        style_bones = torch.cat([style_bones[:,:,:,:2], style_bones[:,:,:,-1:]], dim=-1)
        output_bone_feat = model_bone(style_bones)

        style_images_i = style_image[:, i]  # 该风格的参考图像
        results_i = []

        for j, c_char in enumerate(args.content_char_str):
            content_img = content_image[:, j].to(args.device)
            skeleton_img = skeleton_image[:, j].to(args.device)
            style_images_i = style_images_i.to(args.device)

            from src.iffont.data import valid_characters
            x_ch = valid_characters.test_ch[int(c_char)]
            c_ch = tuple([valid_characters.test_ch[int(k)] for k in args.style_char_str])
            
            adapter = VQAdapter(args.vqgan_path)
            adapter.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            style_image_ori = [Image.open(path).convert('RGB')
                        for path in args.style_image_path[i]]  ### style i, n refs
            noresize_transform = transforms.Compose(
                    [transforms.Resize((128, 128), 
                                    interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])])
            style_image_ori = np.array([noresize_transform(image)
                                 for image in style_image_ori])
            c_idx = np.array([adapter.encode(img).cpu()
                            for img in style_image_ori])  ### style i, n refs
            c_idx = torch.as_tensor(c_idx, dtype=torch.long).to(args.device)
            # c_idx = c_idx.unsqueeze(0).unsqueeze(0)  # old 1 ref
            c_idx = c_idx.unsqueeze(0)  ### n refs
            # print("--------------- c_idx shape: ", c_idx.shape)

            # print("content_img shape: ", content_img.shape)
            # print("style_images_i shape: ", style_images_i.shape)
            # print("skeleton_img shape: ", skeleton_img.shape)
            # print("output_bone_feat shape: ", output_bone_feat.shape)
            # print("x_ch: ", x_ch)
            # print("c_ch: ", c_ch)
            # print("c_idx shape: ", c_idx.shape)

            images = pipe.generate(
                content_images=content_img,
                style_images=style_images_i,
                skeleton_images=skeleton_img,
                bone_feature=output_bone_feat,
                x_ch=x_ch,
                c_ch=c_ch,
                c_idx=c_idx,
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
                correcting_x0_fn=args.correcting_x0_fn,
            )
            results_i.append(images[0])
        images_all.append(results_i)

    # 保存多风格×多内容图像矩阵
    save_generated_image(save_dir=args.save_image_dir,
        images=images_all,
        content_num=len(args.content_char_str),
        resolution=args.resolution)
    ##################################
    end = time.time()
    print(f"Finish the sampling process, costing time {end - start}s")
    
    return images_all












    '''
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
    
    ### component
    from src.iffont.data import valid_characters
    # corpus = valid_characters.valid_ch
    corpus_test = valid_characters.test_ch
    x_ch: str = corpus_test[int(args.content_char)]        
    # c_ch = tuple(corpus_test[int(args.style_char)])
    c_ch = tuple([corpus_test[int(char)] for char in args.style_char])  ### n refs
    
    adapter = VQAdapter(args.vqgan_path)
    adapter.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # style_image_ori = Image.open(args.style_image_path).convert('RGB')
    # style_image_ori = [Image.open(path).convert('RGB') for path in args.style_image_path]  ### n refs
    style_image_ori = [[Image.open(path).convert('RGB')
                        for path in args.style_image_path[i]]
                        for i in range(len(args.style_image_path))]  ### all styles, n refs
    noresize_transform = transforms.Compose(
            [transforms.Resize((128, 128), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    # style_image_ori = noresize_transform(style_image_ori)[None, :]
    # c_idx = adapter.encode(style_image_ori)
    # style_image_ori = np.array([noresize_transform(image) for image in style_image_ori])  ### n refs
    # c_idx = np.array([adapter.encode(img).cpu() for img in style_image_ori])  ### n refs
    style_image_ori = np.array([[noresize_transform(image)
                                 for image in style_image_ori[i]]
                                 for i in range(len(style_image_ori))])  ### all styles, n refs
    c_idx = np.array([[adapter.encode(img).cpu()
                       for img in style_image_ori[i]]
                       for i in range(len(style_image_ori))])  ### all styles, n refs
    c_idx = torch.as_tensor(c_idx, dtype=torch.long).to(args.device)
    # c_idx = c_idx.unsqueeze(0).unsqueeze(0)  # old 1 ref
    c_idx = c_idx.unsqueeze(0)  ### n refs
    # print("--------------- c_idx shape: ", c_idx.shape)
    ###################################################
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        skeleton_image = skeleton_image.to(args.device)
        ##################################
        # model_bone = seq_branch(args)
        model_bone = style_seq_encoder(args)
        model_bone.load_state_dict(torch.load(f"{args.ckpt_dir}/model_bone.pth"))
        model_bone.to(args.device)
        # output_bone = model_bone(content_bones, style_bones, target_bones, device=0)
        ##################################
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        ##################################
        # print("style path: ", np.array(args.style_image_path)[:,0])
        # print("target path: ", args.target_image_path)
        # print("style_image shape: ", style_image.shape)  # [1, 34, 3, 3, 96, 96]
        images_all = []
        for i in range(style_image.shape[1]):
            #################
            style_bone_fonti = []
            style_fonts = sorted(os.listdir(args.style_image_root))
            for style_char_i in style_char:
                style_bone_i = np.load(os.path.join(args.bone_root, style_fonts[i], '{}_{}_bone_sdf.npy'.format(style_fonts[i], style_char_i)))
                if style_bone_i.shape[0] < args.max_bone_len:
                    style_bone_i = np.vstack([style_bone_i, np.zeros((args.max_bone_len-style_bone_i.shape[0], style_bone_i.shape[1]))])
                style_bone_fonti.append(style_bone_i)
            style_bones = torch.FloatTensor(np.array(style_bone_fonti)).unsqueeze(0).to(args.device)   # [1, refs, N,12]

            # target_bone_i = np.load(os.path.join(args.bone_root, style_fonts[i], '{}_{}_bone_sdf.npy'.format(style_fonts[i], content_char)))
            # if target_bone_i.shape[0] < args.max_bone_len:
            #     target_bone_i = np.vstack([target_bone_i, np.zeros((args.max_bone_len-target_bone_i.shape[0], target_bone_i.shape[1]))])
            # target_bones = torch.FloatTensor(np.array(target_bone_i)).unsqueeze(0).to(args.device)   # [1, N,12]
            
            ##################### exclude sdf !!!!!!!!!!!!!!!
            style_bones = torch.cat([style_bones[:,:,:,:2], style_bones[:,:,:,-1:]], dim=-1)
            # content_bones = torch.cat([content_bones[:,:,:2], content_bones[:,:,-1:]], dim=-1)
            # target_bones = torch.cat([target_bones[:,:,:2], target_bones[:,:,-1:]], dim=-1)
            #####################

            # output_bone = model_bone(content_bones, style_bones, target_bones, device=args.device)
            output_bone_style_feat = model_bone(style_bones)
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
            images = pipe.generate(
                content_images=content_image,
                style_images=style_image[:,i],
                skeleton_images=skeleton_image,
                # seq_data=seq_data,  #####
                # feat_data=feat_data,  #####
                # bone_feature=output_bone['seq_feat'],  #####
                bone_feature=output_bone_style_feat,  #####
                x_ch=x_ch,  c_ch=c_ch,  c_idx=c_idx[:,i],  #####
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
    '''


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
    pipe = load_fontdiffuer_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)
