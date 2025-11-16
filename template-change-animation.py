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

# --- 1. Configuration (Please Edit) ---

# Directory containing your .obj files.
OBJ_DIR = Path('log/CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67')

# Directory to store intermediate PNG frames
PNG_DIR = Path('temp')

# Final GIF file path
GIF_PATH = 'animation_pytorch3d.gif'

# File path for the composite PNG
COMPOSITE_PNG_PATH = 'composite_epochs_captioned.png' # Changed name to reflect captioning

# GIF frames per second
GIF_FPS = 10

# Image size (Height, Width)
IMAGE_SIZE = (512, 512)

# --- NEW: Captioning Configuration ---
FONT_PATH = "arial.ttf" # <-- IMPORTANT: Change this to a valid font path on your system!
# Example for Windows: "C:/Windows/Fonts/arial.ttf"
# Example for macOS: "/System/Library/Fonts/Arial.ttf"
# Example for Linux: "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 40
TEXT_COLOR = (255, 255, 255) # White color for text (R, G, B)
TEXT_POSITION_OFFSET = (10, 10) # Offset from top-left corner (x, y)

# --- 2. Setup Device (GPU if available) ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("WARNING: No GPU found. Rendering will be slow.")

# --- 3. Setup Pytorch3D Renderer ---

# --- Camera Setup ---
R, T = look_at_view_transform(dist=2.7, elev=30, azim=0) 
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

# Define targets for the composite PNG
TARGET_EPOCHS = ['010', '020', '040', '080', '160']
TARGET_STEMS = [f'epoch_{epoch}_template' for epoch in TARGET_EPOCHS]
composite_images_map = {}

# --- NEW: Load font for captions ---
try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    print(f"WARNING: Could not load font from {FONT_PATH}. Using default font. Please check FONT_PATH.")
    font = ImageFont.load_default()

# --- 5. Render Loop (OBJ to PNG) ---
for i, obj_file_path in enumerate(tqdm(obj_files, desc="Rendering Frames")):
    
    # 1. Load the OBJ file
    verts, faces, aux = load_obj(obj_file_path, device=device)
    
    # 2. Create a Textures object (using white vertex colors)
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # 3. Create a Meshes object
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.verts_idx.to(device)],
        textures=textures
    )

    # 4. Render the mesh
    try:
        images = renderer(mesh)
        
        # 5. Convert and Save Image
        output_image = images[0, ..., :3].cpu().numpy()
        output_image_uint8 = (output_image * 255).astype('uint8')
        
        # Check if this is one of our target frames
        file_stem = obj_file_path.stem
        
        # Extract epoch number from stem (e.g., 'epoch_010_template' -> '010')
        epoch_str = file_stem.replace('epoch_', '').replace('_template', '')
        
        # --- NEW: Add caption to the image for composite PNG ---
        if epoch_str in TARGET_EPOCHS:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(output_image_uint8)
            draw = ImageDraw.Draw(pil_image)
            # Add text caption
            draw.text(TEXT_POSITION_OFFSET, f"epoch{epoch_str}", font=font, fill=TEXT_COLOR)
            # Store the captioned PIL Image back into the map
            composite_images_map[file_stem] = np.array(pil_image) # Convert back to numpy array
            
        # Save the temporary PNG for the GIF (you might want to caption these too, but current request is for composite)
        output_png_path = PNG_DIR / f'frame_{i:04d}.png'
        imageio.imwrite(output_png_path, output_image_uint8)
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
print(f"\nCreating captioned composite PNG at {COMPOSITE_PNG_PATH}...")

images_to_stack = []
missing_epochs = []

for stem in TARGET_STEMS:
    if stem in composite_images_map:
        images_to_stack.append(composite_images_map[stem])
    else:
        missing_epochs.append(stem)

if missing_epochs:
    print(f"WARNING: Could not find rendered images for all target epochs.")
    print(f"Missing: {', '.join(missing_epochs)}")

if images_to_stack:
    try:
        # Use numpy.hstack (horizontal stack) to join images that now include captions
        composite_image = np.hstack(images_to_stack)
        # Save the composite image
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
