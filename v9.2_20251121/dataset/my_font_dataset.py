import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np
from src.embedding.svg.transformers import numericalize
from src import VQAdapter
from src.embedding.component.data import valid_characters

def get_nonorm_transform(resolution):
    nonorm_transform =  transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                               interpolation=transforms.InterpolationMode.BILINEAR), 
             transforms.ToTensor()])
    return nonorm_transform

def get_noresize_transform(resolution):
    noresize_transform = transforms.Compose(
            [transforms.Resize((resolution, resolution), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    return noresize_transform


class FontDataset(Dataset):
    """The dataset of font generation  
    """
    def __init__(self, args, phase, transforms=None, scr=False):
        super().__init__()
        self.args = args  ###
        self.img_root = args.img_root  ###
        self.svg_root = args.svg_root  ###
        self.skeleton_root = args.skeleton_root  ###
        self.skeleton_img_root = args.skeleton_img_root  ###
        self.feat_root = args.my_feat_root  ###
        # self.ttf_root = args.ttf_root  ###
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = args.num_neg
        
        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        #################### iffont
        self.corpus = valid_characters.valid_ch
        self.corpus_seen = valid_characters.train_ch
        # self.fonts_path = []
        # for f in os.listdir(self.ttf_root):
        #     path = os.path.join(self.ttf_root, f)
        #     if not os.path.isfile(path):
        #         continue
        #     self.fonts_path.append(path)
        self.adapter = VQAdapter(self.args.vqgan_path)
        self.adapter.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.style_vq_transforms = get_noresize_transform(128)

    def get_path(self):
        self.target_images = []
        # images with related style  
        self.style_to_images = {}
        # target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        target_image_dir = f"{self.img_root}"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style
        self.num_fonts = len(os.listdir(target_image_dir))  ###
        self.num_chars = len(self.target_images) // self.num_fonts  ###

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        # target_image_name = target_image_path.split('/')[-1]
        # style, content = target_image_name.split('.')[0].split('+')
        style, content = target_image_path.split('.')[0].split('/')[-2:]
        
        # Read content image
        # content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.jpg"
        content_image_path = f"{self.img_root}/011/{content}.png"  # 新书宋
        content_image = Image.open(content_image_path).convert('RGB')
        if self.args.use_controlnet:
            skeleton_image_path = f"{self.skeleton_img_root}/011/{content}.png"  # skeleton img for controlnet
            skeleton_image = Image.open(skeleton_image_path).convert('RGB')
        else:
            skeleton_image = None

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        # style_image_path = random.choice(images_related_style)
        # style_image = Image.open(style_image_path).convert("RGB")
        style_image_path = random.sample(images_related_style, self.args.num_refs)    ##### 1 ref -> n ref
        style_image = [Image.open(path).convert("RGB") for path in style_image_path]
        style_image_ori = style_image.copy()  #####
        
        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)
        target_image_grey = Image.open(target_image_path).convert("L")
        nonorm_target_image_grey = self.nonorm_transforms(target_image_grey).squeeze(0)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            # style_image = self.transforms[1](style_image)
            style_image = torch.stack([self.transforms[1](image) for image in style_image], dim=0)  ### 1 ref -> n ref
            target_image = self.transforms[2](target_image)
            if self.args.use_controlnet:
                skeleton_image = self.transforms[0](skeleton_image)
        
        ### Read component, self added, like iffont
        # style_font = int(style.split('_')[1])  # 0
        # style_font = '{n:0{w}}'.format(n=style_font, w=len(str(self.num_fonts)))  # 000, all 212
        style_font = style   # already rename
        # style_char = style_image_path.split('.')[0].split('/')[-1]  # random char
        # style_char = '{n:0{w}}'.format(n=int(style_char), w=len(str(self.num_chars)))  # random char -> xxx, all 800
        style_char = [path.split('.')[0].split('/')[-1] for path in style_image_path]  ### n refs
        style_char = ['{n:0{w}}'.format(n=int(style_char_i), w=len(str(self.num_chars))) for style_char_i in style_char]
        content_font = '{n:0{w}}'.format(n=11, w=len(str(self.num_fonts)))  # 011, 新书宋
        content_char = '{n:0{w}}'.format(n=int(content), w=len(str(self.num_chars)))  # 0100 -> 100, all 800
        
        if self.args.use_svg:
            # 1 style svg now, 3 refs to do later
            if os.path.exists(os.path.join(self.svg_root, style_font, '{}_{}_seq_len.npy'.format(style_font, style_char[0]))):
                style_svg_len = torch.LongTensor(np.load(os.path.join(self.svg_root, style_font,  # [1]
                            '{}_{}_seq_len.npy'.format(style_font, style_char[0]))))
                style_svg = torch.FloatTensor(np.load(os.path.join(self.svg_root, style_font, # [261, 12]
                            '{}_{}_sequence_relaxed.npy'.format(style_font, style_char[0]))))
                # 检查SVG数据的维度，确保是二维数组
                if len(style_svg.shape) == 1:
                    style_svg = style_svg.reshape(-1, 1)  # 转换为二维数组
                
                if style_svg.shape[0] < self.args.max_svg_len:
                    style_svg = torch.cat([style_svg, torch.zeros((self.args.max_svg_len-style_svg.shape[0], style_svg.shape[1]))], dim=0)
            else:
                style_svg_len = torch.LongTensor([0])
                style_svg = torch.FloatTensor(np.zeros((self.args.max_svg_len, self.args.dim_svg)))
            # content_svg_len = torch.LongTensor(np.load(os.path.join(self.svg_root, content_font,  # [1]
            #             '{}_{}_seq_len.npy'.format(content_font, content_char))))
            # content_svg = torch.FloatTensor(np.load(os.path.join(self.svg_root, content_font, # [261, 12]
            #             '{}_{}_sequence_relaxed.npy'.format(content_font, content_char))))
            # target_svg_len = torch.LongTensor(np.load(os.path.join(self.svg_root, style_font,  # [1]
            #             '{}_{}_seq_len.npy'.format(style_font, content_char))))
            # target_svg = torch.FloatTensor(np.load(os.path.join(self.svg_root, style_font, # [261, 12]
            #             '{}_{}_sequence_relaxed.npy'.format(style_font, content_char))))
            # style_thickthin = torch.FloatTensor(np.load(os.path.join(self.feat_root, style_font, # [2]
            #             '{}_{}_thickthin.npy'.format(style_font, style_char))))
            
            arg_quant = numericalize(style_svg[:, 4:])
            cmd_cls = torch.argmax(style_svg[:, :4], dim=-1).unsqueeze(-1)
            style_svg = torch.cat([cmd_cls, arg_quant], dim=-1) # 1 + 8 = 9 dimension, 12->9
            # arg_quant = numericalize(content_svg[:, 4:])
            # cmd_cls = torch.argmax(content_svg[:, :4], dim=-1).unsqueeze(-1)
            # content_svg = torch.cat([cmd_cls, arg_quant], dim=-1)
            # # mask
            style_svg_mask = torch.zeros(1, self.args.svg_emb_len)  # svg_emb_len要能被max_svg_len整除
            if style_svg_len[0] > 0:
                denom = self.args.max_svg_len / self.args.svg_emb_len *1.0
                style_svg_mask[:, :int(np.ceil( (style_svg_len[0]+1)/denom ))] = 1
            # content_svg_mask = torch.zeros(1, self.args.svg_emb_len) # [1,9] value = 1 means pos to be masked
            # content_svg_mask[:, :int(np.ceil( (content_svg_len[0]+1)/denom ))] = 1

        # skeleton
        if self.args.use_skeleton:
            # style_skeleton = torch.FloatTensor(np.load(os.path.join(self.skeleton_root, style_font, # [12]
            #             '{}_{}_bone_sdf.npy'.format(style_font, style_char))))
            style_skeleton = []
            # 确定统一的列数，根据日志中的debug信息，应该是3列
            target_columns = 3
            
            for style_char_i in style_char:
                style_skeleton_i = np.load(os.path.join(self.skeleton_root, style_font, '{}_{}_bone.npy'.format(style_font, style_char_i)))
                
                # 检查骨架数据的维度，确保是二维数组
                if len(style_skeleton_i.shape) == 1:
                    style_skeleton_i = style_skeleton_i.reshape(-1, 1)  # 转换为二维数组
                
                # 确保列数一致
                if style_skeleton_i.shape[1] != target_columns:
                    # 如果列数少于目标列数，进行扩展
                    if style_skeleton_i.shape[1] < target_columns:
                        padding = np.zeros((style_skeleton_i.shape[0], target_columns - style_skeleton_i.shape[1]))
                        style_skeleton_i = np.hstack([style_skeleton_i, padding])
                    # 如果列数多于目标列数，进行截断
                    else:
                        style_skeleton_i = style_skeleton_i[:, :target_columns]
                
                # 确保行数一致
                if style_skeleton_i.shape[0] < self.args.max_skeleton_len:
                    style_skeleton_i = np.vstack([style_skeleton_i, np.zeros((self.args.max_skeleton_len - style_skeleton_i.shape[0], target_columns))])
                elif style_skeleton_i.shape[0] > self.args.max_skeleton_len:
                    style_skeleton_i = style_skeleton_i[:self.args.max_skeleton_len, :]
                
                style_skeleton.append(style_skeleton_i)
            
            # 现在所有骨架数据的形状都是(max_skeleton_len, target_columns)
            style_skeleton = torch.FloatTensor(np.array(style_skeleton))   # [refs, N, 3]

            # content_skeleton = torch.FloatTensor(np.load(os.path.join(self.skeleton_root, content_font, # [N,12]
            #             '{}_{}_bone.npy'.format(content_font, content_char))))
            # # 检查骨架数据的维度，确保是二维数组
            # if len(content_skeleton.shape) == 1:
            #     content_skeleton = content_skeleton.reshape(-1, 1)  # 转换为二维数组
            # target_skeleton = torch.FloatTensor(np.load(os.path.join(self.skeleton_root, style_font, # [N,12]
            #             '{}_{}_bone.npy'.format(style_font, content_char))))
            # # 检查骨架数据的维度，确保是二维数组
            # if len(target_skeleton.shape) == 1:
            #     target_skeleton = target_skeleton.reshape(-1, 1)  # 转换为二维数组
            # if content_skeleton.shape[0] < self.args.max_skeleton_len:
            #     content_skeleton = torch.cat([content_skeleton, torch.zeros((self.args.max_skeleton_len-content_skeleton.shape[0], content_skeleton.shape[1]))], dim=0)
            # if target_skeleton.shape[0] < self.args.max_skeleton_len:
            #     target_skeleton = torch.cat([target_skeleton, torch.zeros((self.args.max_skeleton_len-target_skeleton.shape[0], target_skeleton.shape[1]))], dim=0)
            # # style_skeleton = style_skeleton.unsqueeze(0)

            ##################### exclude sdf !!!!!!!!!!!!!!!
            # 由于我们已经将数据统一为3列，这里可能不再需要额外的处理
            # 但为了保持与原有逻辑的兼容性，我们仍然保留这行代码
            style_skeleton = torch.cat([style_skeleton[:,:,:2], style_skeleton[:,:,-1:]], dim=-1)
            # content_skeleton = torch.cat([content_skeleton[:,:2], content_skeleton[:,-1:]], dim=-1)
            # target_skeleton = torch.cat([target_skeleton[:,:2], target_skeleton[:,-1:]], dim=-1)
            #####################

        # component id
        if self.args.use_component:
            # from src.iffont.data.adapter import VQAdapter
            # from src.iffont.data.adapter import pil_to_tensor
            # from src.iffont.util import utils
            # from PIL import ImageFont
            # import warnings

            x_ch: str = self.corpus[int(content)]        
            # # c_ch = tuple(random.sample(self.corpus_seen, k=self.args.num_refs))
            # c_ch = self.corpus_seen[int(style_char)]
            c_ch = tuple([self.corpus_seen[int(style_char_i)] for style_char_i in style_char])  ### 1 ref -> n ref

            # def _synthesis_img(ch, font):
            #     img = utils.draw_single_char(ch, font, self.args.resolution, mode='RGB')
            #     warnings.filterwarnings("ignore", category=UserWarning)
            #     return pil_to_tensor(img)
            # # ch = self.corpus[int(style_char)]
            # # font = self.fonts[font_id]
            # font = ImageFont.truetype(self.fonts_path[int(style)], self.args.resolution)
            # img = _synthesis_img(c_ch, font)
            
            ## c_idx = adapter.encode(img)  #.cpu().numpy()
            # style_image_ori = self.style_vq_transforms(style_image_ori)
            # c_idx = self.adapter.encode(style_image_ori)
            style_image_ori = np.array([self.style_vq_transforms(image) for image in style_image_ori])  ### 1 ref -> n ref
            c_idx = np.array([self.adapter.encode(img).cpu() for img in style_image_ori])
            ## c_idx = self.adapter.encode(torch.zeros((3,128,128)))
            c_idx = torch.as_tensor(c_idx, dtype=torch.long)
            # c_idx = c_idx.unsqueeze(0)  ##### old when not set n ref
            # print("c_idx shape: ", c_idx.shape)  # [num_refs, 256]
            # print("style_image_ori shape: ", style_image_ori.shape)  # [num_refs, 3, 128, 128]


        
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "skeleton_image": skeleton_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
            "nonorm_target_image_grey": nonorm_target_image_grey,
            # "style_svg": style_svg,     ###
            # # "content_svg": content_svg,     ###
            # "style_svg_mask": style_svg_mask,  ###
            # # "content_svg_mask": content_svg_mask,  ###
            # # "target_svg": target_svg,  ###
            # # "target_svg_len": target_svg_len,  ###
            # # "style_thickthin": style_thickthin  ###
            # "style_skeleton": style_skeleton,  ###
            # # "content_skeleton": content_skeleton,  ###
            # # "target_skeleton": target_skeleton,  ###
            # "c_idx": c_idx,
            # "x_ch": x_ch,
            # "c_ch": c_ch,
            }
        if self.args.use_component:
            sample["c_idx"] = c_idx
            sample["x_ch"] = x_ch
            sample["c_ch"] = c_ch
        if self.args.use_skeleton:
            sample["style_skeleton"] = style_skeleton
            # sample["content_skeleton"] = content_skeleton
            # sample["target_skeleton"] = target_skeleton
        if self.args.use_svg:
            sample["style_svg"] = style_svg
            # sample["content_svg"] = content_svg
            sample["style_svg_mask"] = style_svg_mask
            # sample["content_svg_mask"] = content_svg_mask
            # sample["target_svg"] = target_svg
            # sample["target_svg_len"] = target_svg_len
            # sample["style_thickthin"] = style_thickthin
        if self.args.use_controlnet:
            sample["skeleton_image"] = skeleton_image
        

        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                # choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.jpg"
                choose_neg_name = f"{self.img_root}/{choose_style}/{content}.png"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)
