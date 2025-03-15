import tkinter as tk
from tkinter import ttk, messagebox
import torch
import torch.nn as nn
from DCGAN import Generator
from torchvision.utils import save_image
from PIL import Image, ImageTk
import os
import time

class ImageGeneratorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("鸟类图片生成器")
        self.root.geometry("800x600")
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure('Custom.TButton', padding=10, font=('微软雅黑', 12))
        self.style.configure('Custom.TLabel', font=('微软雅黑', 12))
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="AI鸟类图片生成器", 
                               font=('微软雅黑', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=20)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        
        # 生成数量选择
        ttk.Label(control_frame, text="生成数量:").grid(row=0, column=0, padx=5, pady=5)
        self.num_images = tk.StringVar(value="1")
        num_spinbox = ttk.Spinbox(control_frame, from_=1, to=10, 
                                 textvariable=self.num_images, width=10)
        num_spinbox.grid(row=0, column=1, padx=5, pady=5)
        
        # 生成按钮
        self.generate_btn = ttk.Button(control_frame, text="生成图片", 
                                     command=self.generate_images, style='Custom.TButton')
        self.generate_btn.grid(row=1, column=0, columnspan=2, pady=10)
        
        # 进度条
        self.progress = ttk.Progressbar(control_frame, length=200, mode='determinate')
        self.progress.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 状态标签
        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=2, pady=5)
        
        # 图片显示区域
        self.image_frame = ttk.LabelFrame(main_frame, text="生成结果", padding="10")
        self.image_frame.grid(row=1, column=1, padx=10, pady=10, sticky=tk.NSEW)
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
    def load_model(self):
        """加载模型"""
        try:
            num_epochs = 200
            self.nz = 100
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model_path = f'checkpoint/netG{num_epochs}.pth'
            
            if not os.path.exists(model_path):
                messagebox.showerror("错误", f"模型文件 {model_path} 不存在！")
                return
            
            self.netG = torch.load(model_path, map_location=self.device)
            self.netG.eval()
            self.status_var.set(f"模型加载成功 (使用{self.device})")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{str(e)}")
            self.status_var.set("模型加载失败")
    
    def generate_images(self):
        """生成图片"""
        try:
            num_images = int(self.num_images.get())
            self.generate_btn.state(['disabled'])
            self.status_var.set("正在生成...")
            self.progress['value'] = 0
            
            # 创建保存目录
            save_dir = "generated_images"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成图片
            for i in range(num_images):
                noise = torch.randn(1, self.nz, 1, 1, device=self.device)
                with torch.no_grad():
                    fake_image = self.netG(noise)
                
                # 保存图片
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f'generated_{timestamp}_{i+1}.png')
                save_image(fake_image, save_path, normalize=True)
                
                # 显示最新生成的图片
                self.display_image(save_path)
                
                # 更新进度
                self.progress['value'] = (i + 1) / num_images * 100
                self.root.update()
            
            self.status_var.set(f"成功生成 {num_images} 张图片")
            messagebox.showinfo("成功", f"已生成 {num_images} 张图片并保存到 {save_dir} 目录")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成失败：{str(e)}")
            self.status_var.set("生成失败")
        finally:
            self.generate_btn.state(['!disabled'])
    
    def display_image(self, image_path):
        """在界面上显示图片"""
        try:
            # 读取并调整图片大小
            image = Image.open(image_path)
            image = image.resize((300, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            
            # 更新显示
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # 保持引用
        except Exception as e:
            print(f"显示图片失败：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageGeneratorApp(root)
    root.mainloop()