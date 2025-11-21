import os
import argparse
import logging
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (
    FontDiffuserModel,
    build_unet,
    build_style_encoder,
    build_content_encoder,
    build_ddpm_scheduler,
    FontDiffuserDPMPipeline
)
from utils import save_image_with_content_style
from .utils_fid import FIDEvaluator

logger = get_logger(__name__)

def main():
    args = get_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = f"fid_evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/eval_samples", exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        filename=f"{args.output_dir}/fid_evaluation.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    
    # 初始化加速器
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.output_dir)
    
    # 加载模型
    logging.info(f"Loading model from {args.checkpoint_path}")
    
    # 构建模型组件
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    
    # 加载权重
    if os.path.isdir(args.checkpoint_path):
        # 如果是目录，假设包含各个组件的权重
        unet.load_state_dict(torch.load(f"{args.checkpoint_path}/unet.pth", map_location="cpu"))
        style_encoder.load_state_dict(torch.load(f"{args.checkpoint_path}/style_encoder.pth", map_location="cpu"))
        content_encoder.load_state_dict(torch.load(f"{args.checkpoint_path}/content_encoder.pth", map_location="cpu"))
    else:
        # 如果是单个文件，尝试加载整个模型
        try:
            model_state = torch.load(args.checkpoint_path, map_location="cpu")
            if isinstance(model_state, FontDiffuserModel):
                model = model_state
            else:
                # 尝试从字典中加载
                unet.load_state_dict(model_state.get('unet', model_state))
                style_encoder.load_state_dict(model_state.get('style_encoder', style_encoder.state_dict()))
                content_encoder.load_state_dict(model_state.get('content_encoder', content_encoder.state_dict()))
                model = FontDiffuserModel(
                    unet=unet,
                    style_encoder=style_encoder,
                    content_encoder=content_encoder)
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    # 如果没有直接加载到model，则创建
    if 'model' not in locals():
        model = FontDiffuserModel(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder)
    
    # 准备模型
    model = accelerator.prepare(model)
    model.eval()
    
    # 初始化FID评估器
    fid_evaluator = FIDEvaluator(
        device=accelerator.device,
        image_size=args.resolution,
        num_workers=args.dataloader_num_workers
    )
    
    # 创建DPM采样管道
    pipeline = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=noise_scheduler
    )
    pipeline.to(accelerator.device)
    
    # 创建评估数据集
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
    
    # 创建评估数据集和数据加载器
    eval_font_dataset = FontDataset(
        args=args,
        phase='eval', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_font_dataset, 
        shuffle=False, 
        batch_size=args.eval_batch_size, 
        collate_fn=CollateFN(),
        num_workers=args.dataloader_num_workers)
    
    # 执行FID评估
    run_fid_evaluation(
        pipeline=pipeline,
        dataloader=eval_dataloader,
        fid_evaluator=fid_evaluator,
        args=args,
        accelerator=accelerator
    )
    

def run_fid_evaluation(pipeline, dataloader, fid_evaluator, args, accelerator):
    """
    执行FID评估
    
    Args:
        pipeline: FontDiffuserDPMPipeline实例
        dataloader: 评估数据加载器
        fid_evaluator: FIDEvaluator实例
        args: 命令行参数
        accelerator: Accelerator实例
    """
    logging.info(f"Starting FID evaluation with {args.num_eval_samples} samples")
    
    # 重置FID评估器
    fid_evaluator.reset()
    
    # 限制评估样本数量
    num_eval_samples = min(args.num_eval_samples, len(dataloader.dataset))
    sample_count = 0
    
    # 进度条
    progress_bar = tqdm(total=num_eval_samples, desc="FID Evaluation")
    
    with torch.no_grad():
        for samples in dataloader:
            if sample_count >= num_eval_samples:
                break
                
            content_images = samples["content_image"].to(accelerator.device)
            style_images = samples["style_image"].to(accelerator.device)
            target_images = samples["target_image"].to(accelerator.device)
            nonorm_target_images = samples["nonorm_target_image"].to(accelerator.device)
            
            # 生成样本
            try:
                generated_images = pipeline.generate(
                    content_images=content_images,
                    style_images=style_images,
                    bone_feature=None,
                    steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    batch_size=content_images.shape[0]
                )
            except Exception as e:
                logging.error(f"Image generation failed: {str(e)}")
                continue
                
            # 添加真实图像和生成图像到FID评估器
            fid_evaluator.add_real_images_batch(nonorm_target_images)
            fid_evaluator.add_fake_images_batch(generated_images)
            
            # 保存一些评估样本
            if sample_count < args.num_visualize_samples:
                save_dir = f"{args.output_dir}/eval_samples"
                os.makedirs(save_dir, exist_ok=True)
                for i in range(min(4, content_images.shape[0])):  # 每组保存前4个样本
                    save_image_with_content_style(
                        content_image=content_images[i],
                        style_image=style_images[i],
                        gen_image=generated_images[i],
                        target_image=nonorm_target_images[i],
                        save_path=f"{save_dir}/sample_{sample_count}_{i}.png",
                        content_size=args.content_image_size,
                        style_size=args.style_image_size,
                        target_size=(args.resolution, args.resolution)
                    )
            
            # 更新计数和进度条
            current_batch_size = content_images.shape[0]
            sample_count += current_batch_size
            progress_bar.update(current_batch_size)
    
    progress_bar.close()
    
    # 计算FID分数
    try:
        fid_score = fid_evaluator.compute_fid()
        logging.info(f"FID score: {fid_score}")
        print(f"FID score: {fid_score}")
        
        # 保存FID分数到文件
        with open(f"{args.output_dir}/fid_score.txt", "w") as f:
            f.write(f"FID score: {fid_score}\n")
            f.write(f"Number of samples: {sample_count}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Sampling steps: {args.num_inference_steps}\n")
            f.write(f"Guidance scale: {args.guidance_scale}\n")
    except Exception as e:
        logging.error(f"Failed to compute FID score: {str(e)}")
        raise
    
    logging.info("FID evaluation completed")


def get_args():
    parser = get_parser()
    
    # 添加评估特定的参数
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the model checkpoint (directory or file)")
    parser.add_argument("--eval_batch_size", type=int, default=8, 
                        help="Batch size for evaluation")
    parser.add_argument("--num_eval_samples", type=int, default=500, 
                        help="Number of samples to use for FID evaluation")
    parser.add_argument("--num_visualize_samples", type=int, default=10, 
                        help="Number of samples to visualize")
    
    args = parser.parse_args()
    
    # 确保必要的参数有合理的值
    if args.output_dir is None:
        args.output_dir = f"fid_evaluation_{time.strftime('%Y%m%d_%H%M%S')}"
    
    return args


if __name__ == "__main__":
    main()