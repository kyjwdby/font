import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

# 导入单独的模态损失函数
from ..modules.losses.multimodal_loss import SVGModalLoss, SkeletonModalLoss
# 导入骨架解码器
from ..embedding.skeleton.skeleton_model import SkeletonDecoder



class FontDiffuserModel(ModelMixin, ConfigMixin):
    """Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """

    @register_to_config
    def __init__(
        self, 
        unet, 
        controlnet,
        style_encoder,
        content_encoder,
        style_modulator,
        use_multimodal_loss: bool = False,
        svg_encoder: Optional[nn.Module] = None,
        skeleton_encoder: Optional[nn.Module] = None,
        svg_weight: float = 1.0,
        skeleton_weight: float = 1.0,
        use_unet_pre_loss: bool = True,
        use_unet_post_loss: bool = False,
        use_unet_post_skeleton_loss: bool = False,
        skeleton_weight_post: float = 0.5,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_modulator = style_modulator
        self.svg_encoder = svg_encoder
        self.skeleton_encoder = skeleton_encoder
        self.use_multimodal_loss = use_multimodal_loss
        self.use_unet_pre_loss = use_unet_pre_loss
        self.use_unet_post_loss = use_unet_post_loss
        self.use_unet_post_skeleton_loss = use_unet_post_skeleton_loss
        self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.Tensor([1 / 0.07])))
        
        # 初始化骨架解码器（强制输入维度=1024，与style_seq_encoder的hidden_size一致）
        self.skeleton_decoder = SkeletonDecoder(hidden_size=1024, output_n_points=255) if skeleton_encoder is not None else None
        
        # 初始化模态损失函数
        if use_multimodal_loss or use_unet_pre_loss or use_unet_post_loss or use_unet_post_skeleton_loss:
            self.svg_loss = SVGModalLoss()
            self.skeleton_loss = SkeletonModalLoss(
                weight_point=1.0,
                weight_chamfer=2.5,
                weight_structure=1.5
            )
        
        # 初始化损失权重参数
        self.svg_weight = svg_weight
        self.skeleton_weight = skeleton_weight
        self.skeleton_weight_post = skeleton_weight_post
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        style_images,
        content_images,
        skeleton_images,
        content_encoder_downsample_size,
        style_feature_cond,
        skeleton_data: Optional[Dict] = None,  # 添加骨架数据参数
        cond: Optional[Dict] = None,  # 修复：添加条件字典参数
    ):
        # style_img_feature, _, _ = self.style_encoder(style_images)
    
        # batch_size, channel, height, width = style_img_feature.shape
        # style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        # # style_hidden_states = fused_style_feature  # modified: [B, 1024]
        # style_img_feature, style_reduce_feature, _, _ = self.style_encoder(style_images)
        # style_reduce_feature = style_reduce_feature.unsqueeze(1)
        # # style_reduce_feature = style_reduce_feature.unsqueeze(1).repeat(1,3,1)
        
        B, n_refs, _, _, _ = style_images.shape  ##### 1 ref -> n ref
        style_images = style_images.view(B*n_refs, style_images.size(2), style_images.size(3), style_images.size(4))
        style_img_feature, _, _, _ = self.style_encoder(style_images)
        style_img_feature = style_img_feature.view(B, n_refs, style_img_feature.size(1), style_img_feature.size(2), style_img_feature.size(3))
        style_img_feature = style_img_feature.mean(dim=1)
        batch_size, channel, height, width = style_img_feature.shape
        style_img_feature = style_img_feature.reshape(batch_size, channel, 1, height*width)
        ###
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)

        # B, n_refs = style_images.shape[0], 1
        # style_images = style_images[:,0,...]
        # style_img_feature, _, _, _ = self.style_encoder(style_images)

        style_hidden_states_extra = style_feature_cond
        # print("----------------style_hidden_states_extra shape: ", style_hidden_states_extra.shape)  # [B, 35, 1024]
        
        ### img and other modal cat, and prepare InfoNCE loss 
        #########################
        if style_hidden_states_extra is not None:
            style_hidden_states = torch.cat([style_hidden_states, style_hidden_states_extra], dim=1)
            style_img_feature = torch.cat([style_img_feature, style_hidden_states_extra.unsqueeze(-1).permute(0,2,3,1)], dim=3)
        #########################


        # Get the content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)  # [B,3,96,96]
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        ############################ 1 ref -> n ref
        style_content_feature = style_content_feature.view(B, n_refs, 
                                                    style_content_feature.size(1), style_content_feature.size(2), style_content_feature.size(3))
        style_content_feature = style_content_feature.mean(dim=1)
        # print("style_content_feature shape: ", style_content_feature.shape)
        for i in range(len(style_content_res_features)):
            style_content_res_features[i] = style_content_res_features[i].view(B, n_refs, 
                    style_content_res_features[i].size(1), style_content_res_features[i].size(2), style_content_res_features[i].size(3))
            style_content_res_features[i] = style_content_res_features[i].mean(dim=1)
            # print("feature shape: ", feature.shape)
        ############################
        style_content_res_features.append(style_content_feature)  # [B,3,96,96]

        input_hidden_states = [style_img_feature, content_residual_features, \
                               style_hidden_states, style_content_res_features]

        # 初始化模态损失字典（普通版本）
        modal_loss_dict = {}

        # 骨架损失预处理和前向传播（在UNet之前）
        skeleton_feature_extra = None
        if (self.use_unet_pre_loss or self.use_multimodal_loss) and skeleton_data is not None and self.skeleton_decoder is not None:
            # 修复：正确处理骨架编码特征
            # 如果有骨架编码器，使用骨架编码特征；否则退回到style_hidden_states_extra
            if self.skeleton_encoder is not None and 'skeleton_encoder_feat' in cond and cond['skeleton_encoder_feat'] is not None:
                skeleton_coords_pred = self.skeleton_decoder(cond['skeleton_encoder_feat'])
                skeleton_feature_extra = skeleton_coords_pred
                print(f"[DEBUG] FontDiffuserModel: 使用骨架编码特征，形状: {cond['skeleton_encoder_feat'].shape}")
            elif style_hidden_states_extra is not None:
                # 原来的逻辑：当没有骨架编码特征时，使用style_hidden_states_extra
                skeleton_coords_pred = self.skeleton_decoder(style_hidden_states_extra)
                skeleton_feature_extra = skeleton_coords_pred
                print(f"[DEBUG] FontDiffuserModel: 使用style_hidden_states_extra，形状: {style_hidden_states_extra.shape}")
            else:
                print("[WARNING] FontDiffuserModel: 没有可用的骨架特征")
                skeleton_coords_pred = None
                
                # 计算UNet前骨架损失
                if self.use_unet_pre_loss and skeleton_data.get('skeleton_extra', None) is not None:
                    # 对骨架预测添加噪声并计算损失
                    skeleton_extra_gt = skeleton_data['skeleton_extra']  # 获取真实骨架坐标
                    skeleton_extra_noisy = skeleton_extra_gt + torch.randn_like(skeleton_extra_gt) * 0.1  # 添加噪声
                    
                    # 使用完整的骨架损失函数（与DPM版本保持一致）
                    skeleton_loss_dict = self.skeleton_loss(skeleton_coords_pred, skeleton_extra_noisy)
                    modal_loss_dict['skeleton_pre'] = self.skeleton_weight * skeleton_loss_dict['skeleton_total_loss']
                
                # 添加骨架特征到输入隐状态（如果启用预处理损失）
                if self.use_unet_pre_loss and skeleton_feature_extra is not None:
                    # 将骨架特征扩展并与style_hidden_states_extra拼接
                    skeleton_expanded = skeleton_feature_extra.unsqueeze(1).expand(-1, style_hidden_states_extra.size(1), -1)
                    updated_style_hidden_states = torch.cat([style_hidden_states, skeleton_expanded], dim=1)
                    updated_style_img_feature = torch.cat([style_img_feature, skeleton_feature_extra.unsqueeze(-1).permute(0,2,3,1)], dim=3)
                    input_hidden_states = [updated_style_img_feature, content_residual_features, updated_style_hidden_states, style_content_res_features]

        # print("content_img_feature shaoe: ", content_img_feature.shape)
        # print("style_img_feature shape: ", style_img_feature.shape)
        # print("style_hidden_states shape: ", style_hidden_states.shape)
        
        ### controlnet
        # expected_blocks = self.controlnet.diagnose_controlnet_blocks()  ### deepseek unused
        # controlnet_hint = preprocess(control_image).to(dtype=unet.dtype)
        # controlnet_hint = content_images
        if self.controlnet is not None:
            controlnet_hint = skeleton_images
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x_t,
                timesteps,
                encoder_hidden_states=input_hidden_states,
                controlnet_cond=controlnet_hint,
            )
            weights = self.style_modulator(style_img_feature)
            # print("len down_block_res_samples: ", len(down_block_res_samples))
            # print("len mid_block_res_sample: ", len(mid_block_res_sample))
            mod_down = [d * weights[:, int(i)].view(-1, 1, 1, 1) for i, d in enumerate(down_block_res_samples)]
            mod_mid = mid_block_res_sample * weights[:, -1].view(-1, 1, 1, 1)
        else:
            mod_down = None
            mod_mid = None
        ### unet down and mid sample shape
        # [0] shape = (4, 64, 48, 48)
        # [1] shape = (4, 128, 24, 24)
        # [2] shape = (4, 256, 12, 12)
        # [3] shape = (4, 512, 12, 12)
        # mid shape = (4, 512, 12, 12)
        
        ### controlnet output, origin
        # for i, d in enumerate(down_block_res_samples):
        #     print(f"[{i}] shape = {tuple(d.shape)}")
        #     # [0] shape = (4, 64, 96, 96)
        #     # [1] shape = (4, 64, 96, 96)
        #     # [2] shape = (4, 64, 96, 96)
        #     # [3] shape = (4, 64, 48, 48)
        #     # [4] shape = (4, 128, 48, 48)
        #     # [5] shape = (4, 128, 48, 48)
        #     # [6] shape = (4, 128, 24, 24)
        #     # [7] shape = (4, 256, 24, 24)
        #     # [8] shape = (4, 256, 24, 24)
        #     # [9] shape = (4, 256, 12, 12)
        #     # [10] shape = (4, 512, 12, 12)
        #     # [11] shape = (4, 512, 12, 12)
        # for i, d in enumerate(mid_block_res_sample):
        #     print(f"[{i}] shape = {tuple(d.shape)}")
        #     # [0] shape = (512, 12, 12)
        #     # [1] shape = (512, 12, 12)
        #     # [2] shape = (512, 12, 12)
        #     # [3] shape = (512, 12, 12)
        # mod_mid = mid_block_res_sample * weights[:, -1].view(-1, 1, 1, 1)
        # if isinstance(mid_block_res_sample, (list, tuple)):
        #     mod_mid = [m * weights[:, -1-i].view(-1, 1, 1, 1) for i, m in enumerate(mid_block_res_sample)]
        # else:
        #     mod_mid = mid_block_res_sample * weights[:, -1].view(-1, 1, 1, 1)
        
        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            down_block_additional_residuals=mod_down,  ###
            mid_block_additional_residual=mod_mid,  ###
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]

        # 骨架损失后向传播（在UNet之后，普通版本）
        # 注意：这里的后向处理可能需要在外部调用中处理
        # 修复：统一返回4个值以保持与DPM版本的一致性
        modal_loss_dict = {}  # 普通版本的modal_loss_dict为空，但仍需返回以保持接口一致
        return noise_pred, offset_out_sum, style_img_feature, modal_loss_dict


class FontDiffuserModelDPM(ModelMixin, ConfigMixin):
    """DPM Forward function for FontDiffuer with content encoder \
        style encoder and unet.
    """
    @register_to_config
    def __init__(
        self, 
        unet, 
        controlnet,
        style_encoder,
        content_encoder,
        style_modulator,
        use_multimodal_loss: bool = False,
        svg_encoder: Optional[nn.Module] = None,
        skeleton_encoder: Optional[nn.Module] = None,
        svg_weight: float = 1.0,
        skeleton_weight: float = 1.0,
        use_unet_pre_loss: bool = True,
        use_unet_post_loss: bool = False,
        use_unet_post_skeleton_loss: bool = False,
        skeleton_weight_post: float = 0.5,
    ):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.style_encoder = style_encoder
        self.content_encoder = content_encoder
        self.style_modulator = style_modulator
        self.svg_encoder = svg_encoder
        self.skeleton_encoder = skeleton_encoder
        self.use_multimodal_loss = use_multimodal_loss
        self.use_unet_pre_loss = use_unet_pre_loss
        self.use_unet_post_loss = use_unet_post_loss
        self.use_unet_post_skeleton_loss = use_unet_post_skeleton_loss
        self.svg_weight = svg_weight
        self.skeleton_weight = skeleton_weight
        self.skeleton_weight_post = skeleton_weight_post
        self.temp = nn.Parameter(torch.ones([]) * torch.log(torch.Tensor([1 / 0.07])))
        
        # 初始化骨架解码器（强制输入维度=1024，与style_seq_encoder的hidden_size一致）
        self.skeleton_decoder = SkeletonDecoder(hidden_size=1024, output_n_points=76) if skeleton_encoder is not None else None
        
        # 初始化模态损失函数
        if use_multimodal_loss or use_unet_pre_loss or use_unet_post_loss or use_unet_post_skeleton_loss:
            self.svg_loss = SVGModalLoss()
            self.skeleton_loss = SkeletonModalLoss(
                weight_point=1.0,
                weight_chamfer=2.5,
                weight_structure=1.5
            )
    
    def forward(
        self, 
        x_t, 
        timesteps, 
        cond,
        content_encoder_downsample_size,
        version,
        skeleton_data: Optional[Dict] = None,
        svg_data: Optional[Dict] = None,
    ):
        content_images = cond[0]
        style_images = cond[1]
        
        # 检查cond列表长度，安全地获取骨架图像和样式特征
        skeleton_images = cond[2] if len(cond) > 2 else None
        style_feature_cond = cond[3] if len(cond) > 3 else None
        
        # # seq_data = cond[2]
        # bone_feature = cond[3]
        # x_ch = cond[4]
        # c_ch = cond[5]
        # c_idx = cond[6]

        # style_img_feature, _, style_residual_features = self.style_encoder(style_images)
        
        ## batch_size, channel, height, width = style_img_feature.shape
        ## style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        #style_img_feature, style_reduce_feature, _, _ = self.style_encoder(style_images)
        #style_reduce_feature = style_reduce_feature.unsqueeze(1)
        B, n_refs, _, _, _ = style_images.shape  ##### 1 ref -> n ref
        style_images = style_images.view(B*n_refs, style_images.size(2), style_images.size(3), style_images.size(4))
        style_img_feature, _, _, _ = self.style_encoder(style_images)
        style_img_feature = style_img_feature.view(B, n_refs, style_img_feature.size(1), style_img_feature.size(2), style_img_feature.size(3))
        style_img_feature = style_img_feature.mean(dim=1)
        batch_size, channel, height, width = style_img_feature.shape
        style_img_feature = style_img_feature.reshape(batch_size, channel, 1, height*width)
        ###
        style_hidden_states = style_img_feature.permute(0, 2, 3, 1).reshape(batch_size, height*width, channel)
        
        #########################
        style_hidden_states_extra = style_feature_cond
        # print("----------------style_hidden_states_extra shape: ", style_hidden_states_extra.shape)  # [B, 35, 1024]
        #########################
        if style_hidden_states_extra is not None:
            # style_feat = {"style_img_feat": style_hidden_states, "style_fusion_feat": style_hidden_states_extra, "temp": self.temp}
            # print("style_hidden_states shape: ", style_hidden_states.shape)
            # print("style_hidden_states_extra shape: ", style_hidden_states_extra.shape)
            style_hidden_states = torch.cat([style_hidden_states, style_hidden_states_extra], dim=1)
            style_img_feature = torch.cat([style_img_feature, style_hidden_states_extra.unsqueeze(-1).permute(0,2,3,1)], dim=3)
        # else:
        #     style_feat = {"style_img_feat": style_hidden_states, "temp": self.temp}
        #########################

        # Get content feature
        content_img_feature, content_residual_features = self.content_encoder(content_images)
        content_residual_features.append(content_img_feature)
        # Get the content feature from reference image
        style_content_feature, style_content_res_features = self.content_encoder(style_images)
        ############################ 1 ref -> n ref
        style_content_feature = style_content_feature.view(B, n_refs, 
                                                    style_content_feature.size(1), style_content_feature.size(2), style_content_feature.size(3))
        style_content_feature = style_content_feature.mean(dim=1)
        # print("style_content_feature shape: ", style_content_feature.shape)
        for i in range(len(style_content_res_features)):
            style_content_res_features[i] = style_content_res_features[i].view(B, n_refs, 
                    style_content_res_features[i].size(1), style_content_res_features[i].size(2), style_content_res_features[i].size(3))
            style_content_res_features[i] = style_content_res_features[i].mean(dim=1)
            # print("feature shape: ", feature.shape)
        ############################
        style_content_res_features.append(style_content_feature)

        input_hidden_states = [style_img_feature, content_residual_features, style_hidden_states, style_content_res_features]

        # 初始化模态损失字典
        modal_loss_dict = {}
        
        # 骨架损失预处理和前向传播（在UNet之前）
        skeleton_feature_extra = None
        if (self.use_unet_pre_loss or self.use_multimodal_loss) and skeleton_data is not None and self.skeleton_decoder is not None:
            # 修复：使用正确的风格特征格式作为解码器输入
            skeleton_coords_pred = self.skeleton_decoder(style_hidden_states_extra)
            
            # 骨架数据添加噪声处理
            skeleton_coords_gt = skeleton_data.get('skeleton_extra', None)
            if skeleton_coords_gt is not None:
                # 添加时间步噪声
                noise = torch.randn_like(skeleton_coords_gt) * 0.1
                skeleton_coords_noisy = skeleton_coords_gt + noise
                
                # 计算骨架损失（使用完整的骨架损失函数）
                if self.use_unet_pre_loss or self.use_multimodal_loss:
                    skeleton_loss_dict = self.skeleton_loss(skeleton_coords_pred, skeleton_coords_noisy)
                    modal_loss_dict['skeleton_pre'] = self.skeleton_weight * skeleton_loss_dict['skeleton_total_loss']
                    
            skeleton_feature_extra = skeleton_coords_pred  # 作为额外特征

        ### controlnet
        # expected_blocks = self.controlnet.diagnose_controlnet_blocks()  ### deepseek unused
        # controlnet_hint = preprocess(control_image).to(dtype=unet.dtype)
        # controlnet_hint = content_images
        if self.controlnet is not None:
            controlnet_hint = skeleton_images
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x_t,
                timesteps,
                encoder_hidden_states=input_hidden_states,
                controlnet_cond=controlnet_hint,
            )
            weights = self.style_modulator(style_img_feature)
            # print("len down_block_res_samples: ", len(down_block_res_samples))
            # print("len mid_block_res_sample: ", len(mid_block_res_sample))
            mod_down = [d * weights[:, int(i)].view(-1, 1, 1, 1) for i, d in enumerate(down_block_res_samples)]
            mod_mid = mid_block_res_sample * weights[:, -1].view(-1, 1, 1, 1)
        else:
            mod_down = None
            mod_mid = None
        
        
        out = self.unet(
            x_t, 
            timesteps, 
            encoder_hidden_states=input_hidden_states,
            down_block_additional_residuals=mod_down,  ###
            mid_block_additional_residual=mod_mid,  ###
            content_encoder_downsample_size=content_encoder_downsample_size,
        )
        noise_pred = out[0]
        offset_out_sum = out[1]  # 修复：确保返回offset_out_sum
        
        # 骨架损失后向传播（在UNet之后）
        if (self.use_unet_post_loss or self.use_unet_post_skeleton_loss) and skeleton_data is not None:
            # 修复：使用正确的风格特征格式作为解码器输入
            skeleton_coords_pred_post = self.skeleton_decoder(style_hidden_states_extra)
            
            skeleton_coords_gt = skeleton_data.get('skeleton_extra', None)
            if skeleton_coords_gt is not None and self.use_unet_post_skeleton_loss:
                # 计算后向骨架损失（使用完整的骨架损失函数）
                skeleton_loss_dict = self.skeleton_loss(skeleton_coords_pred_post, skeleton_coords_gt)
                modal_loss_dict['skeleton_total_loss_post'] = self.skeleton_weight_post * skeleton_loss_dict['skeleton_total_loss']
        
        # 修复：返回4个值以匹配训练代码期望
        style_feat = style_img_feature  # 使用处理后的特征
        return noise_pred, offset_out_sum, style_feat, modal_loss_dict
