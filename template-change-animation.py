import torch
import imageio
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np  # <-- 1. Added numpy

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
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

# <-- 2. NEW: File path for the composite PNG -->
COMPOSITE_PNG_PATH = 'composite_epochs.png'

# GIF frames per second
GIF_FPS = 10

# Image size (Height, Width)
IMAGE_SIZE = (512, 512)

# --- 2. Setup Device (GPU if available) ---
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    print("WARNING: No GPU found. Rendering will be slow.")

# --- 3. Setup Pytorch3D Renderer ---

# --- Camera Setup ---
# (dist=distance, elev=elevation, azim=azimuth)
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

# --- 5. Render Loop (OBJ to PNG) ---

# <-- 3. Define targets for the composite PNG -->
TARGET_EPOCHS = ['010', '020', '040', '080', '160']
# Assuming file format 'epoch_XXX_template.obj'
TARGET_STEMS = [f'epoch_{epoch}_template' for epoch in TARGET_EPOCHS]
# Use a dictionary to store the image data for specific frames by name
composite_images_map = {}


for i, obj_file_path in enumerate(tqdm(obj_files, desc="Rendering Frames")):
    
    # 1. Load the OBJ file
    verts, faces, aux = load_obj(obj_file_path, device=device)
    
    # 2. Create a Textures object (using white vertex colors)
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
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
        
        # <-- 4. Check if this is one of our target frames -->
        file_stem = obj_file_path.stem
        if file_stem in TARGET_STEMS:
            # If so, store its numpy array in the dictionary
            composite_images_map[file_stem] = output_image_uint8
            
        # 6. Save the temporary PNG for the GIF
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

# --- 6.5. Create Composite PNG (Newly Added Section) ---
print(f"\nCreating composite PNG at {COMPOSITE_PNG_PATH}...")

images_to_stack = []
missing_epochs = []

# Collect images in the order specified by TARGET_STEMS
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
        # Use numpy.hstack (horizontal stack) to join images
        composite_image = np.hstack(images_to_stack)
        # Save the composite image
        imageio.imwrite(COMPOSITE_PNG_PATH, composite_image)
        print(f"Composite PNG saved: {COMPOSITE_PNG_PATH}")
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
