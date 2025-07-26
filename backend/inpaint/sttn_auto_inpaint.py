import os
import copy
import time
import sys
from typing import List

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import get_inpaint_area_by_mask, is_frame_number_in_ab_sections
from backend.tools.subtitle_detect import SubtitleDetect

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

class STTNInpaint:
    def __init__(self, device, model_path):
        self.device = device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        # 根据设备选择合适的map_location
        map_location = 'cpu' if self.device.type == 'cpu' else self.device
        self.model.load_state_dict(torch.load(model_path, map_location=map_location)['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 设置相连帧数
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = int(W_ori * 3 / 16)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, mask)
        # 打印修复区域信息以便调试
        print(f"Inpaint areas: {inpaint_area}")
        print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}")
        # 检查掩码中是否有非零值
        if mask.sum() == 0:
            print("Warning: Mask is empty!")
        # 初始化帧存储变量
        # 高分辨率帧存储列表
        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}  # 存放缩放后帧的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (W_ori, split_h))  # 将补全帧缩放回原大小
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转换颜色空间
                    # 获取遮罩区域并进行图像合成
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]  # 取出遮罩区域
                    # 实现遮罩区域内的图像融合
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        # 确保所有帧都是numpy数组
        for i in range(len(frames)):
            if not isinstance(frames[i], np.ndarray):
                frames[i] = np.array(frames[i])
                
        frame_length = len(frames)
        print(f"Starting inpaint processing for {frame_length} frames")
        
        # 检查CUDA内存
        if torch.cuda.is_available():
            print(f"CUDA memory - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB, "
                  f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
        
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # 把特征张量转移到指定的设备（CPU或GPU）
        feats = feats.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats_reshaped = feats.view(frame_length, 3, self.model_input_height, self.model_input_width)
            feats_reshaped = feats_reshaped.to(self.device)  # 确保在正确的设备上
            feats_encoded = self.model.encoder(feats_reshaped)
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats_encoded.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats_encoded.view(1, frame_length, c, feat_h, feat_w)
        # 获取重绘区域
        # 在设定的邻居帧步幅内循环处理视频
        for f in range(0, frame_length, self.neighbor_stride):
            print(f"Processing batch {f//self.neighbor_stride + 1}/{(frame_length-1)//self.neighbor_stride + 1}")
            # 计算邻近帧的ID
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            # 获取参考帧的索引
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            # 同样关闭梯度计算
            with torch.no_grad():
                # 通过模型推断特征并传递给解码器以生成完成的帧
                feats_input = feats[0, neighbor_ids + ref_ids, :, :, :]
                feats_input = feats_input.to(self.device)  # 确保在正确的设备上
                pred_feat = self.model.infer(feats_input)
                # 将预测的特征通过解码器生成图片，并应用激活函数tanh，然后分离出张量
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                # 确保结果在CPU上以便后续处理
                if self.device.type != 'cpu':
                    pred_img = pred_img.cpu()
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # 将预测的图片转换为无符号8位整数格式
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        # 如果该位置为空，则赋值为新计算出的图片
                        comp_frames[idx] = img
                    else:
                        # 如果此位置之前已有图片，则将新旧图片混合以提高质量
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        print(f"Completed inpaint processing for {frame_length} frames")
        return comp_frames


class STTNAutoInpaint:

    def read_frame_info_from_video(self):
        # 使用opencv读取视频
        reader = cv2.VideoCapture(self.video_path)
        
        # 如果指定了开始时间，设置视频从该时间开始
        if self.start_time > 0:
            reader.set(cv2.CAP_PROP_POS_MSEC, self.start_time * 1000)
        
        # 获取视频的宽度, 高度, 帧率和帧数信息并存储在frame_info字典中
        # 注意：CAP_PROP_FRAME_WIDTH是宽度，CAP_PROP_FRAME_HEIGHT是高度
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # 视频的原始宽度
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # 视频的原始高度
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 视频的帧率
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 视频的总帧数
        }
        
        # 计算实际要处理的帧数
        start_frame = int(self.start_time * frame_info['fps']) if self.start_time > 0 else 0
        end_frame = int(self.end_time * frame_info['fps']) if self.end_time is not None else frame_info['len']
        actual_frame_count = min(end_frame, frame_info['len']) - start_frame
        frame_info['actual_len'] = actual_frame_count
        frame_info['start_frame'] = start_frame
        frame_info['end_frame'] = end_frame
        
        print(f"Frame info: {frame_info}")
        print(f"Calculated frames: start={start_frame}, end={end_frame}, count={actual_frame_count}")
        # 验证尺寸
        print(f"Verified frame dimensions - Width: {frame_info['W_ori']}, Height: {frame_info['H_ori']}")
        
        # 返回视频读取对象、帧信息和视频写入对象
        return reader, frame_info

    def __init__(self, device, model_path, video_path, mask_path=None, clip_gap=None, sub_areas=None, start_time=0, end_time=None):
        # STTNInpaint视频修复实例初始化
        self.sttn_inpaint = STTNInpaint(device, model_path)
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        # 设置输出视频文件的路径，只包含处理的片段
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub_{int(start_time)}s_{int(end_time)}s.mp4"
        ) if end_time is not None else os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # 配置可在一次处理中加载的最大帧数
        if clip_gap is None:
            self.clip_gap = config.getSttnMaxLoadNum()
        else:
            self.clip_gap = clip_gap
            
        # 初始化OCR检测器，用于检测帧中是否包含文字
        self.subtitle_detector = None
        if sub_areas is not None:
            self.subtitle_detector = SubtitleDetect(video_path, sub_areas)
            
        # 视频处理的起始和结束时间
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        print(f"STTNAutoInpaint: {self.start_time}s to {self.end_time}s")  # 简化日志
        reader = None
        writer = None
        try:
            # 读取视频帧信息
            reader, frame_info = self.read_frame_info_from_video()
            print(f"Frame info: {frame_info}")
            if input_sub_remover is not None:
                ab_sections = input_sub_remover.ab_sections
                # 使用input_sub_remover的writer来写入处理过的帧
                writer = input_sub_remover.video_writer
                # 在STTN-AUTO模式下创建一个新的writer用于输出只包含处理片段的视频
                # 注意：尺寸参数是(宽度, 高度)
                standalone_writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
                print(f"Created standalone writer: {self.video_out_path}")
                # 验证writer是否成功创建
                if not standalone_writer.isOpened():
                    print("Error: Failed to create standalone video writer!")
            else:
                ab_sections = None
                # 创建视频写入对象，用于输出修复后的视频
                # 注意：尺寸参数是(宽度, 高度)
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
                standalone_writer = writer
                print(f"Created writer: {self.video_out_path}")
                # 验证writer是否成功创建
                if not writer.isOpened():
                    print("Error: Failed to create video writer!")
            
            # 使用实际帧数而不是总帧数
            total_frames = frame_info.get('actual_len', frame_info['len'])
            start_frame = frame_info.get('start_frame', 0)
            end_frame = frame_info.get('end_frame', frame_info['len'])
            actual_frame_count = frame_info.get('actual_len', frame_info['len'])  # 获取实际帧数
            
            # 计算需要迭代修复视频的次数
            # 确保rec_time计算正确
            rec_time = actual_frame_count // self.clip_gap
            if actual_frame_count % self.clip_gap != 0:
                rec_time += 1
            print(f"Total frames: {actual_frame_count}, Clip gap: {self.clip_gap}, Rec time: {rec_time}")
            
            # 计算分割高度，用于确定修复区域的大小
            split_h = int(frame_info['W_ori'] * 3 / 16)
            
            if input_mask is None:
                # 读取掩码
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
                mask = mask[:, :, None]
                
            # 得到修复区域位置
            inpaint_area = get_inpaint_area_by_mask(frame_info['W_ori'], frame_info['H_ori'], split_h, mask)
            # 打印修复区域信息以便调试
            print(f"Inpaint areas: {inpaint_area}")
            print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum()}")
            # 检查掩码中是否有非零值
            if mask.sum() == 0:
                print("Warning: Mask is empty!")
            # 验证修复区域是否在图像范围内
            for i, area in enumerate(inpaint_area):
                ymin, ymax, xmin, xmax = area
                if ymin < 0 or ymax > frame_info['H_ori'] or xmin < 0 or xmax > frame_info['W_ori']:
                    print(f"Warning: Inpaint area {i} is out of bounds: {area}")
                print(f"Inpaint area {i}: ymin={ymin}, ymax={ymax}, xmin={xmin}, xmax={xmax}")
                print(f"Frame dimensions: height={frame_info['H_ori']}, width={frame_info['W_ori']}")
            # 遍历每一次的迭代次数
            for i in range(rec_time):
                start_f = i * self.clip_gap  # 起始帧位置
                end_f = min((i + 1) * self.clip_gap, actual_frame_count)  # 结束帧位置
                print(f'Processing segment: {start_f + 1} - {end_f} / {actual_frame_count}')
                
                frames_hr = []  # 高分辨率帧列表
                frames = {}  # 帧字典，用于存储裁剪后的图像
                comps = {}  # 组合字典，用于存储修复后的图像
                
                # 初始化帧字典
                for k in range(len(inpaint_area)):
                    frames[k] = []
                    
                # 读取和修复高分辨率帧
                valid_frames_count = 0
                processed_frames_count = 0  # 记录实际处理的帧数
                skipped_frames_count = 0    # 记录跳过的帧数
                # 只在需要时打印详细信息
                if config.skipFramesWithTextInSttnAuto.value:
                    print(f"Reading {end_f - start_f} frames from video")
                for j in range(start_f, end_f):
                    success, image = reader.read()
                    if not success:
                        if config.skipFramesWithTextInSttnAuto.value:
                            print(f"Failed to read frame {j}.")
                        break
                    
                    # 确保image是numpy数组
                    if not isinstance(image, np.ndarray):
                        image = np.array(image)
                    
                    # 检测帧中是否包含文字，如果包含则跳过该帧（仅在启用配置时）
                    contains_text = False
                    if (config.skipFramesWithTextInSttnAuto.value and 
                        self.subtitle_detector is not None and 
                        is_frame_number_in_ab_sections(j + start_frame, ab_sections)):
                        if config.skipFramesWithTextInSttnAuto.value:
                            print(f"Detecting text in frame {j + start_frame}")
                        detected_text = self.subtitle_detector.detect_subtitle(image)
                        contains_text = len(detected_text) > 0
                        if contains_text:
                            skipped_frames_count += 1
                            if config.skipFramesWithTextInSttnAuto.value:
                                print(f"Frame {j + start_frame} contains text, skipping...")
                        else:
                            processed_frames_count += 1
                            if config.skipFramesWithTextInSttnAuto.value:
                                print(f"Frame {j + start_frame} is clean, will be processed")
                    elif is_frame_number_in_ab_sections(j + start_frame, ab_sections):
                        # 即使不检测文字也记录处理的帧
                        processed_frames_count += 1
                        if config.skipFramesWithTextInSttnAuto.value:
                            print(f"Frame {j + start_frame} will be processed (text detection disabled or skipped)")
                    elif config.skipFramesWithTextInSttnAuto.value:
                        print(f"Frame {j + start_frame} is outside processing area, skipping...")
                    
                    frames_hr.append(image)
                    valid_frames_count += 1
                    
                    # 只有不包含文字的帧才进行处理
                    if is_frame_number_in_ab_sections(j + start_frame, ab_sections) and not contains_text:
                        for k in range(len(inpaint_area)):
                            # 裁剪、缩放并添加到帧字典
                            # 注意：inpaint_area的格式是(ymin, ymax, xmin, xmax)
                            image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 正确使用y坐标
                            # 减少打印频率
                            if config.skipFramesWithTextInSttnAuto.value and j % 10 == 0:
                                print(f"Frame {j + start_frame}: Cropped region shape: {image_crop.shape}")
                            image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                            frames[k].append(image_resize)
                
                print(f"Finished reading frames. Valid: {valid_frames_count}, Processed: {processed_frames_count}, Skipped: {skipped_frames_count}")
                
                # 如果没有读取到有效帧，则跳过当前迭代
                if valid_frames_count == 0:
                    print(f"Skipped segment {start_f+1}-{end_f}")
                    continue
                    
                # 对每个修复区域运行修复
                for k in range(len(inpaint_area)):
                    if len(frames[k]) > 0:  # 确保有帧可以处理
                        # 减少打印频率
                        if config.skipFramesWithTextInSttnAuto.value:
                            print(f"Running inpaint for area {k} with {len(frames[k])} frames")
                        comps[k] = self.sttn_inpaint.inpaint(frames[k])
                        if config.skipFramesWithTextInSttnAuto.value:
                            print(f"Completed processing area {k}")
                    else:
                        comps[k] = []
                        if config.skipFramesWithTextInSttnAuto.value:
                            print(f"Skipping area {k} - no frames to process")
                
                # 如果有要修复的区域
                if inpaint_area and valid_frames_count > 0:
                    # 创建一个映射，记录哪些帧被处理了以及它们在frames[k]中的索引
                    processed_frames_map = {}
                    processed_idx = 0
                    
                    # 构建映射关系
                    if config.skipFramesWithTextInSttnAuto.value:
                        print(f"Building frame mapping for {valid_frames_count} frames")
                    for j in range(start_f, end_f):
                        if j - start_f < valid_frames_count and is_frame_number_in_ab_sections(j + start_frame, ab_sections):
                            # 检查该帧是否包含文字（仅在启用配置时）
                            contains_text = False
                            if (config.skipFramesWithTextInSttnAuto.value and 
                                self.subtitle_detector is not None):
                                detected_text = self.subtitle_detector.detect_subtitle(frames_hr[j - start_f])
                                contains_text = len(detected_text) > 0
                            
                            # 只有不包含文字的帧才被标记为已处理
                            if not contains_text:
                                processed_frames_map[j - start_f] = processed_idx
                                processed_idx += 1
                    
                    if config.skipFramesWithTextInSttnAuto.value:
                        print(f"Processed frames map: {processed_frames_map}")
                    
                    # 应用修复结果
                    if config.skipFramesWithTextInSttnAuto.value:
                        print(f"Applying results to {valid_frames_count} frames")
                    for j in range(valid_frames_count):
                        absolute_frame_number = j + start_f + start_frame  # 计算绝对帧号
                        if input_sub_remover is not None and input_sub_remover.gui_mode:
                            original_frame = copy.deepcopy(frames_hr[j])
                        else:
                            original_frame = None
                            
                        frame = frames_hr[j]
                        
                        # 只有被处理过的帧才应用修复结果
                        if j in processed_frames_map:
                            comp_idx = processed_frames_map[j]
                            # 在应用修复前检查特定像素
                            # 选择修复区域内的像素点进行检查
                            check_y = (inpaint_area[k][0] + inpaint_area[k][1]) // 2
                            check_x = (inpaint_area[k][2] + inpaint_area[k][3]) // 2  # 修复区域的中心点
                            if check_y < frame.shape[0] and check_x < frame.shape[1]:
                                before_pixel = frame[check_y, check_x].copy()  # 保存修复前的像素值
                                # 只在需要时打印
                                if config.skipFramesWithTextInSttnAuto.value and j % 10 == 0:
                                    print(f"Checking pixel at ({check_y}, {check_x}): {before_pixel}")
                            else:
                                before_pixel = None
                                if config.skipFramesWithTextInSttnAuto.value:
                                    print(f"Pixel check coordinates out of bounds: ({check_y}, {check_x})")
                            for k in range(len(inpaint_area)):
                                if comp_idx < len(comps[k]):  # 确保索引有效
                                    # 将修复的图像重新扩展到原始分辨率，并融合到原始帧
                                    comp = cv2.resize(comps[k][comp_idx], (frame_info['W_ori'], split_h))
                                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                                    # 注意：inpaint_area的格式是(ymin, ymax, xmin, xmax)
                                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3]]  # 正确使用y和x坐标
                                    # 检查是否应该应用修复
                            if mask_area.sum() > 0:  # 只有掩码非空时才应用修复
                                if config.skipFramesWithTextInSttnAuto.value and j % 10 == 0:
                                    print(f"Applying inpaint to area: {inpaint_area[k]}, mask_area sum: {mask_area.sum()}")
                                frame[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], inpaint_area[k][2]:inpaint_area[k][3], :]
                            # 检查修复后的像素值
                            if before_pixel is not None and check_y < frame.shape[0] and check_x < frame.shape[1]:
                                after_pixel = frame[check_y, check_x]
                                # 只有像素值发生变化时才报告
                                if not np.array_equal(before_pixel, after_pixel):
                                    if config.skipFramesWithTextInSttnAuto.value and j % 10 == 0:
                                        print(f"Applied inpainting to frame {absolute_frame_number} - Pixel changed from {before_pixel} to {after_pixel}")
                                elif config.skipFramesWithTextInSttnAuto.value and j % 50 == 0:
                                    print(f"Applied inpainting to frame {absolute_frame_number} - No visible change")
                            elif config.skipFramesWithTextInSttnAuto.value and j % 50 == 0:
                                print(f"Applied inpainting to frame {absolute_frame_number} - Checked")
                        elif config.skipFramesWithTextInSttnAuto.value and j % 50 == 0:
                            print(f"Skipped frame {absolute_frame_number}")
                
                # 写入帧到两个writer（如果它们不同的话）
                # 检查帧尺寸
                expected_shape = (frame_info['H_ori'], frame_info['W_ori'], 3)
                if config.skipFramesWithTextInSttnAuto.value and frame.shape != expected_shape:
                    print(f"Frame shape: {frame.shape}, Expected: {expected_shape}")
                if frame.shape != expected_shape:
                    if config.skipFramesWithTextInSttnAuto.value:
                        print(f"Frame dimensions mismatch! Resizing frame to {frame_info['H_ori']}x{frame_info['W_ori']}")
                    frame = cv2.resize(frame, (frame_info['W_ori'], frame_info['H_ori']))
                
                result1 = writer.write(frame)
                if not result1 and config.skipFramesWithTextInSttnAuto.value:
                    print(f"Error: Failed to write frame {absolute_frame_number} to main writer")
                if standalone_writer != writer:
                    result2 = standalone_writer.write(frame)
                    if not result2 and config.skipFramesWithTextInSttnAuto.value:
                        print(f"Error: Failed to write frame {absolute_frame_number} to standalone writer")
                    if config.skipFramesWithTextInSttnAuto.value:
                        print(f"Wrote frame {absolute_frame_number} to standalone writer, success: {result2}")
                elif config.skipFramesWithTextInSttnAuto.value:
                    print(f"Wrote frame {absolute_frame_number} to writer, success: {result1}")
                
                if input_sub_remover is not None:
                    if tbar is not None:
                        input_sub_remover.update_progress(tbar, increment=1)
                    if original_frame is not None and input_sub_remover.gui_mode:
                        input_sub_remover.update_preview_with_comp(original_frame, frame)
        except Exception as e:
            import traceback
            print(f"Error during video processing: {str(e)}")
            print(traceback.format_exc())
            # 不抛出异常，允许程序继续执行
        finally:
            if writer:
                writer.release()
            if reader:
                reader.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    # 记录开始时间
    start = time.time()
    sttn_video_inpaint = STTNAutoInpaint(video_path, mask_path, clip_gap=config.getSttnMaxLoadNum())
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
