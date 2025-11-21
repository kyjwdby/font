import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

# from dataset.font_dataset import FontDataset
from dataset.my_font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 ContentPerceptualLoss,
                 build_unet,
                 build_controlnet,
                 build_style_encoder,
                 build_content_encoder,
                 build_component_encoder,
                 build_component_fusioner,
                 build_skeleton_encoder,
                 build_svg_encoder,
                 build_svg_decoder,
                 build_ddpm_scheduler,
                 build_scr)
# from src.model_diffuser import FontDiffuserModel  #####
# from src.model_bone import seq_branch, style_seq_encoder  #####
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std)

from src import StyleAttention, StyleModulator, VQAdapter
# from src.iffont.data.adapter import VQAdapter
from src.main.loss import mdnloss, cross_modal_loss_normalized

import multiprocessing

# import numpy as np
# import random
# from PIL import Image

logger = get_logger(__name__)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def prepare_style_feature_cond(args, modules, datas):
    # style_svg = datas["style_svg"]  ###
    # content_svg = datas["content_svg"]  ###
    # style_svg_mask = datas["style_svg_mask"]  ###
    # content_svg_mask = datas["content_svg_mask"]  ###
    # target_svg = datas["target_svg"]  ###
    # target_svg_len = datas["target_svg_len"]  ###
    # style_thickthin = datas["style_thickthin"]  ###
    # target_svg_len = target_svg_len.squeeze()  ##### [B]
    # style_thickthin = style_thickthin.unsqueeze(1)  ##### [B,1,2]

    ### !!!!!!!

    # content_skeletons = datas["content_skeleton"]  ###
    # target_skeletons = datas["target_skeleton"]  ###
    # target_skeletons = torch.cat((torch.zeros_like(target_skeletons[:, 0:1, :]).to(device=0), target_skeletons), dim=1)
    
    # component_encoder = modules.get('component_encoder', None)
    # component_fusioner = modules.get('component_fusioner', None)
    # skeleton_encoder = modules.get('skeleton_encoder', None)
    # svg_encoder = modules.get('svg_encoder', None)
    # style_attn = modules.get('style_attn', None)
    
    if args.use_component:
        component_encoder = modules['component_encoder']
        component_fusioner = modules['component_fusioner']
        x_ch = datas["x_ch"]
        c_ch = datas["c_ch"]
        c_idx = datas["c_idx"]
        ids_embed, ids_embed2 = component_encoder(x_ch)# [B, max_len, n_embd]
        sim = component_encoder.coverage(x_ch, c_ch)  # [B,n]
        fused_component_feat, moco_cl = component_fusioner(c_idx, ids_embed2, sim)
        # print("fused_component_feat: ", fused_component_feat.shape)

    if args.use_skeleton:
        skeleton_encoder = modules['skeleton_encoder']
        # 修复：处理4维骨架数据格式 [B, num_refs, max_skeleton_len, 3]
        if 'style_skeleton' in datas:
            style_skeletons = datas["style_skeleton"]  # [B, num_refs, max_skeleton_len, 3]
            print(f"[DEBUG] prepare_style_feature_cond: style_skeletons shape: {style_skeletons.shape}")
            
            # 处理4维数据格式：[B, num_refs, max_skeleton_len, 3]
            if style_skeletons.ndim == 4 and style_skeletons.shape[-1] == 3:
                # 验证格式正确
                batch_size, num_refs, max_skeleton_len, coord_dim = style_skeletons.shape
                print(f"[DEBUG] style_skeletons dims: B={batch_size}, num_refs={num_refs}, max_skeleton_len={max_skeleton_len}, coord_dim={coord_dim}")
                
                # 处理多参考样本：取第一个参考样本并复制到batch_size
                if num_refs > 1:
                    skeleton_input = style_skeletons[:, 0:1, :, :]  # [B, 1, max_skeleton_len, 3]
                    skeleton_input = skeleton_input.repeat(1, 1, 1, 1)  # 保持单个参考样本
                else:
                    skeleton_input = style_skeletons  # [B, 1, max_skeleton_len, 3]
                
                # 直接传递给骨架编码器（不需要squeeze，保持4维格式[B, N, L, 3]）
                skeleton_encoder = skeleton_encoder.to(skeleton_input.device)
                skeleton_feat = skeleton_encoder(skeleton_input)
                print(f"[DEBUG] skeleton_encoder output: {skeleton_feat.shape}")
            else:
                print(f"[ERROR] Invalid style_skeletons format: {style_skeletons.shape}, expected 4D with last dim=3")
                skeleton_feat = None
        else:
            skeleton_feat = None
            print("[DEBUG] No style_skeleton in datas")

    if args.use_svg:
        svg_encoder = modules['svg_encoder']
        # print("--------------------style_svg: ", datas["style_svg"].shape)  # [B,3500,9]
        # print("--------------------style_svg_mask: ", datas["style_svg_mask"].shape)  # [B,1,9]
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
    # 返回融合特征和骨架编码特征（用于骨架解码器）
    return fused_style_feature, skeleton_feat if args.use_skeleton and skeleton_encoder is not None else (fused_style_feature, None)




def main():

    # multiprocessing.set_start_method('spawn')  ### self !!!!!!!!!
    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        # cpu=True,     ############
        )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Ser training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    if args.use_controlnet:
        controlnet = build_controlnet(args=args, unet=unet)
    else:
        controlnet = None
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    # thickthin_encoder = build_thickthin_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        if args.use_controlnet:
            controlnet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/controlnet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))

    ##################################### multi-modal encoders
    # modules = {}
    modules = nn.ModuleDict()  #####
    # params = []
    # 初始化多模态损失组件
    # 使用骨架损失时需要初始化skeleton_encoder，否则为None
    if hasattr(args, 'use_multimodal_loss') and args.use_multimodal_loss:
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
    else:
        # 保持原有逻辑
        skeleton_encoder = None
        svg_encoder = None
        svg_decoder = None
        
    if args.use_component:
        component_encoder = build_component_encoder(args)
        component_encoder.set_device(accelerator.device)
        component_fusioner = build_component_fusioner(args)
        component_fusioner.set_device(accelerator.device)
        modules['component_encoder'] = component_encoder
        modules['component_fusioner'] = component_fusioner
    else:
        component_encoder = None
        component_fusioner = None
    
    if args.use_skeleton and not (hasattr(args, 'use_multimodal_loss') and args.use_multimodal_loss):
        skeleton_encoder = build_skeleton_encoder(args)
        modules['skeleton_encoder'] = skeleton_encoder
        
    if args.use_svg and not (hasattr(args, 'use_multimodal_loss') and args.use_multimodal_loss):
        svg_encoder = build_svg_encoder(args=args)
        svg_decoder = build_svg_decoder(args=args)
        modules['svg_encoder'] = svg_encoder
        modules['svg_decoder'] = svg_decoder
        
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
        
    # 设置多模态损失参数
    use_multimodal_loss = getattr(args, 'use_multimodal_loss', False)
    use_unet_pre_loss = getattr(args, 'use_unet_pre_loss', True)
    use_unet_post_loss = getattr(args, 'use_unet_post_loss', False)
    use_unet_post_skeleton_loss = getattr(args, 'use_unet_post_skeleton_loss', False)
    
    model_diffuser = FontDiffuserModel(
        unet=unet,
        controlnet=controlnet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        style_modulator=style_modulator,
        # 多模态损失相关参数
        use_multimodal_loss=use_multimodal_loss,
        svg_encoder=svg_encoder,
        skeleton_encoder=skeleton_encoder,
        svg_weight=getattr(args, 'svg_weight', 1.0),
        skeleton_weight=getattr(args, 'skeleton_weight', 1.0),
        use_unet_pre_loss=use_unet_pre_loss,
        use_unet_post_loss=use_unet_post_loss,
        use_unet_post_skeleton_loss=use_unet_post_skeleton_loss,
        skeleton_weight_post=getattr(args, 'skeleton_weight_post', 0.5),
        )
    # # model_bone = seq_branch(args)
    # model_bone = style_seq_encoder(args)
    
    
    # Build content perceptaual Loss
    perceptual_loss = ContentPerceptualLoss()

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # Load the datasets
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())
        # train_font_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=8, collate_fn=CollateFN())
    
    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    # 修复参数组重复问题，只使用model_diffuser的参数
    param_groups = [{"params": model_diffuser.parameters()}]
    # 不需要额外添加model_diffuser参数，已经在上面包含
    optimizer = torch.optim.AdamW(
        # model_diffuser.parameters(),
        # [{'params': model_diffuser.parameters()},  #####
        #  {'params': model_bone.parameters()}
        #  ],
        param_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)
    
    # Accelerate preparation
    # model_diffuser, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    #     model_diffuser, optimizer, train_dataloader, lr_scheduler)
    # model_diffuser, model_bone, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(   #####
    #     model_diffuser, model_bone, optimizer, train_dataloader, lr_scheduler)
    model_diffuser, modules, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(   #####
        model_diffuser, modules, optimizer, train_dataloader, lr_scheduler)
    
    ### resume trainer state (self)
    global_step = 0
    if args.resume_from_checkpoint is not None and os.path.exists(args.resume_from_checkpoint):
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)  # 自动恢复 model, optimizer, lr_scheduler
        # 恢复训练步数
        trainer_step_path = os.path.join(args.resume_from_checkpoint, "trainer_step.pt")
        if os.path.exists(trainer_step_path):
            global_step = torch.load(trainer_step_path)["global_step"]
    
    ## move scr module to the target deivces
    if args.phase_2:
        scr = scr.to(accelerator.device)
    
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    num_train_epochs = math.ceil((args.max_train_steps-global_step) / num_update_steps_per_epoch)  ### self !!!
    
    model_diffuser.train()
    # model_bone.train()   #####
    for model in modules.values():
        model.train()
    # global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            skeleton_images = samples["skeleton_image"]
            nonorm_target_images = samples["nonorm_target_image"]
            nonorm_target_images_grey = samples["nonorm_target_image_grey"]

            # 准备风格特征条件
            # 准备风格特征条件
            style_feature_cond, skeleton_encoder_feat = prepare_style_feature_cond(args, modules, samples)
            
            # 修复：移除未定义的 other_cond 变量，直接准备条件
            # 不需要 other_cond，因为所有参数都是作为关键字参数传递的
            with accelerator.accumulate(model_diffuser):

                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)
                x_t = noisy_target_images

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value==1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # 准备骨架数据（如果有的话）
                skeleton_data = None
                cond = {}  # 修复：创建条件字典以供 model_diffuser 使用
                
                if hasattr(args, 'use_unet_pre_loss') and args.use_unet_pre_loss and 'style_skeleton' in samples:
                    # 修复：格式转换 - 将 [num_refs, max_skeleton_len, 3] 转换为 [batch_size, max_skeleton_len, 3]
                    style_skeleton = samples['style_skeleton']  # [num_refs, 76, 3]
                    batch_size = target_images.shape[0]
                    
                    if style_skeleton.ndim == 3 and style_skeleton.shape[0] > 0:
                        # 多参考样本格式：取第一个参考样本并复制到batch_size
                        skeleton_extra = style_skeleton[0:1].repeat(batch_size, 1, 1)  # [batch_size, 76, 3]
                    elif style_skeleton.ndim == 2:
                        # 单维格式：需要添加batch维度
                        skeleton_extra = style_skeleton.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, 76, 3]
                    else:
                        skeleton_extra = style_skeleton
                    
                    skeleton_data = {
                        'skeleton_extra': skeleton_extra
                    }
                    # 将骨架编码特征添加到条件字典中
                    if skeleton_encoder_feat is not None:
                        cond['skeleton_encoder_feat'] = skeleton_encoder_feat
                
                # 准备SVG数据（如果有的话）
                svg_data = None
                if hasattr(args, 'use_multimodal_loss') and args.use_multimodal_loss and 'svg_data' in samples:
                    svg_data = samples['svg_data']

                # Predict the noise residual and compute loss
                # 修复：统一处理4个返回值以保持与DPM版本的一致性
                noise_pred, offset_out_sum, style_img_feature, modal_loss_dict = model_diffuser(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    skeleton_images=skeleton_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    style_feature_cond=style_feature_cond,
                    skeleton_data=skeleton_data,  # 传递骨架数据
                    cond=cond,  # 修复：传递条件字典
                    )
                
                # 处理模态损失（如果启用）
                # 注意：modal_loss_dict现在直接从model_diffuser返回，无需重复初始化
                # if args.use_unet_pre_loss:
                #     # 这里可以添加普通版本的模态损失计算逻辑
                #     # 由于普通版本不包含复杂的模态损失处理，modal_loss_dict为空
                #     pass
                    
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                # svg_total_loss = model_diffuser.seq_encoder.loss(command_logits, args_logits, target_svg.long(), target_svg_len, trg_pts_aux=None)
                
                # output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                
                # seq_point_loss = mdnloss(target_skeletons[:, 1:], output_skeleton['fake'])

                # skeleton rasterize to img, cross modal loss
                # (pi, mu, sigma, rho, label) = output_skeleton['fake']
                # skeleton_raster_loss, pred_heatmap = cross_modal_loss_normalized(
                #     pi, mu, sigma, rho, nonorm_target_images_grey, H=args.resolution, W=args.resolution, sigma_heatmap=0.03
                # )

                # InfoNCE loss（使用普通版本的返回格式）
                celoss = torch.nn.CrossEntropyLoss()
                gt = torch.arange(args.train_batch_size, dtype=torch.long).to(accelerator.device)
                # 使用style_img_feature作为特征进行InfoNCE计算
                style_img_feat = style_img_feature.mean(dim=[2, 3])  # [B, C, H, W] -> [B, C]
                style_fusion_feat = style_feature_cond.mean(dim=1) if style_feature_cond is not None else style_img_feat  # 使用style_feature_cond或退回到style_img_feat
                # 确保所有张量在同一设备上
                style_img_feat = style_img_feat.to(accelerator.device)
                style_fusion_feat = style_fusion_feat.to(accelerator.device)
                gt = gt.to(style_img_feat.device)
                # 温度参数
                temp = getattr(model_diffuser, 'temp', torch.tensor(1.0, device=accelerator.device)).exp()
                logit = temp * style_img_feat @ style_fusion_feat.t()
                info_nce_loss = celoss(logit, gt) + celoss(logit.t(), gt)
                
                # 计算总损失
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss + \
                          args.contrast_coefficient * info_nce_loss
                
                # 添加模态损失
                for modal_loss_name, modal_loss_value in modal_loss_dict.items():
                    loss += modal_loss_value
                            #   args.skeleton_coefficient * seq_point_loss + \
                        #   args.skeleton_raster_coefficient * skeleton_raster_loss
                            # 0
                            #    args.svg_coefficient * svg_total_loss['loss_total']  #####
                
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    # sc loss
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model_diffuser.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        accelerator.save_state(save_dir)
                        torch.save({"global_step": global_step}, os.path.join(save_dir, "trainer_step.pt"))

                        # torch.save(model_diffuser.unet.state_dict(), f"{save_dir}/unet.pth")
                        # torch.save(model_diffuser.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                        # torch.save(model_diffuser.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                        # # torch.save(model_diffuser, f"{save_dir}/total_model.pth")
                        # if style_attn is not None:
                        #     torch.save(style_attn.state_dict(), f"{save_dir}/style_attn.pth")
                        # if args.use_component:
                        #     torch.save(component_encoder.state_dict(), f"{save_dir}/component_encoder.pth")
                        #     torch.save(component_fusioner.state_dict(), f"{save_dir}/component_fusioner.pth")
                        # if args.use_skeleton:
                        #     torch.save(skeleton_encoder.state_dict(), f"{save_dir}/skeleton_encoder.pth")
                        # if args.use_svg:
                        #     torch.save(svg_encoder.state_dict(), f"{save_dir}/svg_encoder.pth")
                        #     torch.save(svg_decoder.state_dict(), f"{save_dir}/svg_decoder.pth")
                        # if args.use_controlnet:
                        #     torch.save(model_diffuser.controlnet.state_dict(), f"{save_dir}/controlnet.pth")
                        #     torch.save(model_diffuser.style_modulator.state_dict(), f"{save_dir}/style_modulator.pth")
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}")
                        print("Save the checkpoint on global step {}".format(global_step))

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                # logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Global Step {global_step} => train_loss = {loss}")
                logging.info(f"Global Step {global_step} => train_loss={loss:.3f}, "
                             +f"infonce_loss={(args.contrast_coefficient * info_nce_loss):.3f}, percep_loss={(args.perceptual_coefficient * percep_loss):.3f}")
            progress_bar.set_postfix(**logs)
            
            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()
