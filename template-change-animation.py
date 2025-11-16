import torch
import imageio
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont # <-- 1. Import Pillow components
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

# --- 1. Configuration (Please Edit) ---

# 包含 .obj 文件的目录
# OBJ_DIR = Path('log/CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67')
OBJ_DIR = Path('log/CUB_wgan_b48_ic1_hard_bg_L1_ganW0_lr0.7_em7_update-1_chf_lpl_reg0.1_data2_depthC0.1_flat10_drop220_gap2_beta0.95_bn_restart1_contour0.1/')


# 存储中间 PNG 帧的目录
PNG_DIR = Path('temp')

# 最终 GIF 文件路径
GIF_PATH = 'animation_pytorch3d.gif'

# 拼接 PNG 的文件路径
COMPOSITE_PNG_PATH = 'composite_epochs_captioned.png'

# GIF 帧率
GIF_FPS = 10

# 图像尺寸 (高, 宽)
IMAGE_SIZE = (512, 512)

# --- 标题配置 (新) ---
# !! 重要: 修改为你系统上的真实字体路径 !!
FONT_PATH = "arial.ttf" 
# 例如 Windows: "C:/Windows/Fonts/arial.ttf"
# 例如 macOS: "/System/Library/Fonts/Arial.ttf"
# 例如 Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 128
TEXT_COLOR = (0, 0, 0) # (R, G, B) - 黑色
TEXT_MARGIN_TOP = 64   # 标题距离图像顶部的像素

# --- 2. Setup Device (GPU if available) ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("WARNING: No GPU found. Rendering will be slow.")

# --- 3. Setup Pytorch3D Renderer ---

# --- Camera Setup ---
if 'CUB' in OBJ_DIR:
    R, T = look_at_view_transform(dist=2.7, elev=30, azim=0)
else:
    R, T = look_at_view_transform(dist=2.7, elev=30, azim=-45)
cameras = PerspectiveCameras(device=device, R=R, T=T)

# --- Rasterizer Setup ---
raster_settings = RasterizationSettings(
    image_size=IMAGE_SIZE,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# --- Light Setup ---
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

# --- Shader Setup ---
shader = SoftPhongShader(
    device=device,
    cameras=cameras,
    lights=lights
)

# --- Final Renderer ---
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=shader
)

print("Renderer setup complete.")

# --- 4. Find and Sort OBJ Files ---
PNG_DIR.mkdir(exist_ok=True)

obj_files = sorted(OBJ_DIR.glob('epoch_*_template.obj'))

if not obj_files:
    print(f"ERROR: No 'epoch_*_template.obj' files found in {OBJ_DIR}")
    exit()

print(f"Found {len(obj_files)} .obj files. Starting render loop...")

image_filepaths = []

# 定义拼接PNG所需的目标
TARGET_EPOCHS = ['020', '040', '080', '160', '320']
TARGET_STEMS = [f'epoch_{epoch}_template' for epoch in TARGET_EPOCHS]
composite_images_map = {}

# --- 加载字体 ---
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print(f"WARNING: 无法从 {FONT_PATH} 加载字体。将使用默认字体。")
    print(f"请检查 FONT_PATH 变量是否设置正确。")
    font = ImageFont.load_default()

# --- 5. Render Loop (OBJ to PNG) ---
#
# !! 主要修改在此处 !!
#
for i, obj_file_path in enumerate(tqdm(obj_files, desc="Rendering Frames")):
    
    # 1. 加载 OBJ
    verts, faces, aux = load_obj(obj_file_path, device=device)
    
    # 2. 创建 Textures (使用白色顶点色)
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # 3. 创建 Meshes
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.verts_idx.to(device)],
        textures=textures
    )

    # 4. 渲染
    try:
        images = renderer(mesh)
        
        # 5. 转换图像为 uint8 numpy 数组
        output_image = images[0, ..., :3].cpu().numpy()
        output_image_uint8 = (output_image * 255).astype('uint8')
        
        # --- 6. (新) 添加居中标题 ---
        
        # 从文件名提取 epoch 编号 (例如 'epoch_010_template' -> '010')
        file_stem = obj_file_path.stem
        try:
            # 假设格式为 epoch_XXX_...
            epoch_str = file_stem.split('_')[1] 
        except IndexError:
            epoch_str = "???" # 如果命名不规范，则使用 '???'
            
        caption_text = f"epoch{epoch_str}"

        # 将 numpy 数组转换为 PIL 图像以便绘制
        pil_image = Image.fromarray(output_image_uint8)
        draw = ImageDraw.Draw(pil_image)

        # 计算文本宽度以实现居中
        try:
            # 较新版本 Pillow
            text_width = draw.textlength(caption_text, font=font)
        except AttributeError:
            # 兼容旧版本 Pillow
            text_width, _ = font.getsize(caption_text)
            
        image_width, _ = pil_image.size
        
        # 计算 x 坐标
        text_x = (image_width - text_width) / 2
        
        # 在图像上绘制黑色居中标题
        draw.text(
            (text_x, TEXT_MARGIN_TOP), 
            caption_text, 
            font=font, 
            fill=TEXT_COLOR
        )

        # 将带标题的 PIL 图像转换回 numpy 数组
        captioned_image_np = np.array(pil_image)

        # --- 7. 保存图像 ---

        # 检查是否为我们想要的特定帧
        if file_stem in TARGET_STEMS:
            # 存储带标题的图像，用于拼接
            composite_images_map[file_stem] = captioned_image_np
            
        # 保存带标题的临时PNG，用于GIF
        output_png_path = PNG_DIR / f'frame_{i:04d}.png'
        imageio.imwrite(output_png_path, captioned_image_np)
        image_filepaths.append(output_png_path)
        
    except Exception as e:
        print(f"\nError rendering {obj_file_path}: {e}")
        print("This might be an empty or corrupted mesh. Skipping.")


# --- 6. Create GIF ---
print(f"\nRendering complete. Compiling GIF at {GIF_PATH}...")

with imageio.get_writer(GIF_PATH, mode='I', fps=GIF_FPS) as writer:
    for filename in tqdm(image_filepaths, desc="Creating GIF"):
        image = imageio.imread(filename)
        writer.append_data(image)

print(f"GIF saved: {GIF_PATH}")

# --- 6.5. Create Composite PNG (With Captions) ---
#
# !! 这部分被简化了，因为它现在使用的是已经带标题的图片 !!
#
print(f"\nCreating captioned composite PNG at {COMPOSITE_PNG_PATH}...")

images_to_stack = []
missing_epochs = []

for stem in TARGET_STEMS:
    if stem in composite_images_map:
        # composite_images_map[stem] 已经包含标题了
        images_to_stack.append(composite_images_map[stem])
    else:
        missing_epochs.append(stem)

if missing_epochs:
    print(f"WARNING: 未能找到所有目标epoch的渲染结果。")
    print(f"缺失: {', '.join(missing_epochs)}")

if images_to_stack:
    try:
        # 使用 numpy.hstack 水平拼接图像
        composite_image = np.hstack(images_to_stack)
        # 保存组合后的图像
        imageio.imwrite(COMPOSITE_PNG_PATH, composite_image)
        print(f"Captioned Composite PNG saved: {COMPOSITE_PNG_PATH}")
    except Exception as e:
        print(f"Error creating composite image: {e}")
else:
    print("No images found for composite. Skipping.")


# --- 7. (Optional) Cleanup ---
print("\nCleaning up intermediate PNG files...")
for filename in image_filepaths:
    os.remove(filename)
os.rmdir(PNG_DIR)

print("Script finished.")
