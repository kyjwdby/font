import torch
import torchmetrics
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from tqdm import tqdm

class FIDEvaluator:
    """
    FID评估器，用于计算真实图像和生成图像之间的FID分数
    使用torchmetrics的FID实现
    """
    def __init__(self, device='cuda', image_size=64, num_workers=None):
        """
        初始化FID评估器
        
        Args:
            device: 运行设备，默认为'cuda'
            image_size: 图像大小，默认为64
            num_workers: 数据加载器工作进程数（此处保留以保持接口兼容性）
        """
        self.device = device
        self.fid = torchmetrics.image.fid.FrechetInceptionDistance(feature=2048).to(device)
        
        # 图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 用于保存累积的图像特征
        self.real_features = []
        self.fake_features = []
    
    def preprocess_image(self, image):
        """
        预处理单张图像
        
        Args:
            image: PIL图像或numpy数组或tensor
        
        Returns:
            预处理后的tensor图像
        """
        if isinstance(image, torch.Tensor):
            # 如果是tensor，确保形状是[C, H, W]并在0-1范围内
            if image.dim() == 4:
                image = image.squeeze(0)
            if image.max() > 1.0:
                image = image / 255.0
        elif isinstance(image, np.ndarray):
            # 如果是numpy数组，转换为PIL图像
            image = Image.fromarray(image)
        
        # 应用预处理变换
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def add_real_image(self, image):
        """
        添加真实图像到FID评估器
        
        Args:
            image: 真实图像（PIL、numpy或tensor格式）
        """
        preprocessed = self.preprocess_image(image)
        self.fid.update(preprocessed, real=True)
        
    def add_fake_image(self, image):
        """
        添加生成图像到FID评估器
        
        Args:
            image: 生成图像（PIL、numpy或tensor格式）
        """
        preprocessed = self.preprocess_image(image)
        self.fid.update(preprocessed, real=False)
    
    def add_real_images_batch(self, images):
        """
        批量添加真实图像
        
        Args:
            images: 真实图像批次，形状为[B, C, H, W]的tensor
        """
        # 确保图像在0-1范围内
        if images.max() > 1.0:
            images = images / 255.0
        
        # 调整大小
        if (images.shape[2] != self.transform.transforms[0].size[0] or 
            images.shape[3] != self.transform.transforms[0].size[1]):
            resize_transform = transforms.Resize((self.transform.transforms[0].size[0], 
                                                self.transform.transforms[0].size[1]))
            images = resize_transform(images)
        
        # 归一化
        normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        images = normalize_transform(images)
        
        self.fid.update(images.to(self.device), real=True)
    
    def add_fake_images_batch(self, images):
        """
        批量添加生成图像
        
        Args:
            images: 生成图像批次，形状为[B, C, H, W]的tensor
        """
        # 确保图像在0-1范围内
        if images.max() > 1.0:
            images = images / 255.0
        
        # 调整大小
        if (images.shape[2] != self.transform.transforms[0].size[0] or 
            images.shape[3] != self.transform.transforms[0].size[1]):
            resize_transform = transforms.Resize((self.transform.transforms[0].size[0], 
                                                self.transform.transforms[0].size[1]))
            images = resize_transform(images)
        
        # 归一化
        normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        images = normalize_transform(images)
        
        self.fid.update(images.to(self.device), real=False)
    
    def compute_fid(self):
        """
        计算并返回FID分数
        
        Returns:
            FID分数
        """
        return self.fid.compute().item()
    
    def reset(self):
        """
        重置FID评估器
        """
        self.fid.reset()

# 便捷函数：评估模型的FID分数
def evaluate_fid(model, dataloader, num_samples=5000, device='cuda'):
    """
    评估模型的FID分数
    
    Args:
        model: 要评估的生成模型
        dataloader: 包含真实图像的数据加载器
        num_samples: 评估的样本数量
        device: 运行设备
        
    Returns:
        FID分数
    """
    fid_evaluator = FIDEvaluator(device=device)
    
    # 收集真实图像和生成图像
    real_count = 0
    fake_count = 0
    
    with torch.no_grad():
        # 收集真实图像
        for batch in tqdm(dataloader, desc="收集真实图像"):
            real_images = batch['image'].to(device) if isinstance(batch, dict) else batch[0].to(device)
            fid_evaluator.add_real_images_batch(real_images)
            real_count += real_images.shape[0]
            if real_count >= num_samples:
                break
        
        # 生成假图像
        model.eval()
        while fake_count < num_samples:
            # 根据模型类型不同，生成图像的方式可能不同
            # 这里假设模型有sample方法
            fake_images = model.sample(16, device=device)
            fid_evaluator.add_fake_images_batch(fake_images)
            fake_count += fake_images.shape[0]
    
    # 计算FID分数
    fid_score = fid_evaluator.compute_fid()
    return fid_score

# 便捷函数：将FID评估集成到训练流程中
def integrate_fid_into_training(train_loader, val_loader, model, output_dir, 
                              eval_interval=5000, device='cuda'):
    """
    将FID评估集成到训练流程中
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model: 训练的模型
        output_dir: 输出目录，用于保存FID结果
        eval_interval: 评估间隔（迭代次数）
        device: 运行设备
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    def eval_callback(step):
        if step % eval_interval == 0:
            print(f"\n评估FID分数 (步骤 {step})...")
            fid_score = evaluate_fid(model, val_loader, device=device)
            print(f"步骤 {step} 的FID分数: {fid_score:.4f}")
            
            # 保存FID结果
            with open(os.path.join(output_dir, 'fid_scores.txt'), 'a') as f:
                f.write(f"{step},{fid_score:.4f}\n")
    
    return eval_callback