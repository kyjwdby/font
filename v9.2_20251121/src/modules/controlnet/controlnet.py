# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)
from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.utils import BaseOutput, logging

from ..unet.embeddings import TimestepEmbedding, Timesteps
from ..unet.unet_blocks import (DownBlock2D,
                          UNetMidMCABlock2D,
                          get_down_block,
                          )
from ..unet.unet import UNet


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNet(ModelMixin, ConfigMixin, FromOriginalControlnetMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = None,
        up_block_types: Tuple[str] = None,
        block_out_channels: Tuple[int] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 64*16,
        attention_head_dim: int = 1,
        channel_attn: bool = True,
        content_encoder_downsample_size: int = 3,
        content_start_channel: int = 64,
        reduction: int = 32,
        
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        conditioning_scale: float = 1.0,
        global_pool_conditions: bool = False,
    ):
        super().__init__()

        self.content_encoder_downsample_size = content_encoder_downsample_size

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )

        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        
        # down
        output_channel = block_out_channels[0]
        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)

        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if i != 0: 
                content_channel = content_start_channel * (2 ** (i-1))
            else:
                content_channel = 0

            print("Load the down block ", down_block_type)
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                content_channel=content_channel,
                reduction=reduction,
                channel_attn=channel_attn,
            )
            self.down_blocks.append(down_block)

            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        self.mid_block = UNetMidMCABlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            channel_attn=channel_attn,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            content_channel=content_start_channel*(2**(content_encoder_downsample_size - 1)),
            reduction=reduction,
        )
        
        mid_block_channel = block_out_channels[-1]

        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block


    @classmethod
    def from_unet(
        cls,
        unet: UNet,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        conditioning_scale: float = 1.0,
        load_weights_from_unet: bool = True,
    ):

        controlnet = cls(
            sample_size=unet.config.sample_size,
            in_channels=unet.config.in_channels,
            out_channels=unet.config.out_channels,
            flip_sin_to_cos=unet.config.flip_sin_to_cos,
            freq_shift=unet.config.freq_shift,
            down_block_types=unet.config.down_block_types,
            up_block_types=unet.config.up_block_types,
            block_out_channels=unet.config.block_out_channels,
            layers_per_block=unet.config.layers_per_block,
            downsample_padding=unet.config.downsample_padding,
            mid_block_scale_factor=unet.config.mid_block_scale_factor,
            act_fn=unet.config.act_fn,
            norm_num_groups=unet.config.norm_num_groups,
            norm_eps=unet.config.norm_eps,
            cross_attention_dim=unet.config.cross_attention_dim,
            attention_head_dim=unet.config.attention_head_dim,
            channel_attn=unet.config.channel_attn,
            content_encoder_downsample_size=unet.config.content_encoder_downsample_size,
            content_start_channel=unet.config.content_start_channel,
            reduction=unet.config.reduction,
            
            conditioning_channels=conditioning_channels,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            conditioning_scale=conditioning_scale,
        )

        if load_weights_from_unet:
            controlnet.conv_in.load_state_dict(unet.conv_in.state_dict())
            controlnet.time_proj.load_state_dict(unet.time_proj.state_dict())
            controlnet.time_embedding.load_state_dict(unet.time_embedding.state_dict())

            controlnet.down_blocks.load_state_dict(unet.down_blocks.state_dict())
            controlnet.mid_block.load_state_dict(unet.mid_block.state_dict())

        return controlnet

    
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attention_slice
    def set_attention_slice(self, slice_size):
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (DownBlock2D)):
            module.gradient_checkpointing = value

    # def diagnose_controlnet_blocks(self):
    #     """诊断 ControlNet blocks 配置"""
    #     print("=== ControlNet Blocks Diagnosis ===")
    #     print(f"Number of down_blocks: {len(self.down_blocks)}")
    #     print(f"Number of controlnet_down_blocks: {len(self.controlnet_down_blocks)}")
        
    #     # 模拟前向传播以确定实际的 down_block_res_samples 数量
    #     with torch.no_grad():
    #         dummy_sample = torch.randn(1, 3, 64, 64).to(0)
    #         dummy_timestep = torch.tensor([0.5]).to(0)
    #         dummy_encoder = torch.randn(1, 77, 256).to(0)
    #         dummy_cond = torch.randn(1, 3, 64, 64).to(0)
            
    #         # 只运行到 down blocks 部分
    #         sample = self.conv_in(dummy_sample)
    #         controlnet_cond = self.controlnet_cond_embedding(dummy_cond)
    #         sample = sample + controlnet_cond
            
    #         down_block_res_samples = (sample,)
    #         for downsample_block in self.down_blocks:
    #             sample, res_samples = downsample_block(hidden_states=sample, temb=torch.randn(1, 256).to(0), index=4)
    #             down_block_res_samples += res_samples
            
    #         print(f"Expected down_block_res_samples length: {len(down_block_res_samples)}")
    #         print(f"ControlNet blocks available: {len(self.controlnet_down_blocks)}")
            
    #         if len(down_block_res_samples) != len(self.controlnet_down_blocks):
    #             print(f"MISMATCH: Need {len(down_block_res_samples)} blocks but have {len(self.controlnet_down_blocks)}")
    #         else:
    #             print("Blocks count matches")
        
    #     return len(down_block_res_samples)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        timestep_cond: Optional[torch.Tensor] = None,
        guess_mode: bool = False,
        content_encoder_downsample_size: int = 4,
        return_dict: bool = False,
    ) -> Union[ControlNetOutput, Tuple]:
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        # emb = self.time_embedding(t_emb, timestep_cond)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # print("controlnet_cond shape: ", controlnet_cond.shape)  # [4, 3, 96, 96]
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        # print("sample shape: ", sample.shape)  # [4, 64, 96, 96]
        # print("controlnet_cond shape: ", controlnet_cond.shape)  # [4, 64, 96, 96]
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = (sample,)
        for index, downsample_block in enumerate(self.down_blocks):
            if (hasattr(downsample_block, "attentions") and downsample_block.attentions is not None) or hasattr(downsample_block, "content_attentions"):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    index=index,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                index=content_encoder_downsample_size,
                encoder_hidden_states=encoder_hidden_states,
            )
        ####
        # for i, d in enumerate(down_block_res_samples):
        #     print(f"[{i}] shape = {tuple(d.shape)}")
        if len(down_block_res_samples) != len(self.controlnet_down_blocks):
            raise RuntimeError(f"down_block_res_samples len {len(down_block_res_samples)} != controlnet_down_blocks len {len(self.controlnet_down_blocks)}")
        # 5. Control net blocks

        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * self.config.conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * self.config.conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * self.config.conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        ############## self: align with unet
        selected_indices = [3, 6, 9, 11]
        down_block_res_samples = [down_block_res_samples[i] for i in selected_indices]
        mid_block_res_sample = mid_block_res_sample[-1]
        ##############

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
