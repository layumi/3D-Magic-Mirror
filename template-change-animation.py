import torch
import imageio
from pathlib import Path
import os
from tqdm import tqdm

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

# Directory containing your .obj files. You need to retrain the model.
OBJ_DIR = Path('log/CamN2_MKT_wgan_b48_lr0.5_em7_update-1_lpl_reg0.1_data2_m2_flat20_depthR0.15_drop220_gap2_beta0.95_clean67')

# Directory to store intermediate PNG frames
PNG_DIR = Path('temp')

# Final GIF file path
GIF_PATH = 'animation_pytorch3d.gif'

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
# We need a fixed camera for all frames.
# You may need to adjust 'dist', 'elev', 'azim' to get the best view.
# 'dist' = distance from origin
# 'elev' = elevation angle (up/down)
# 'azim' = azimuth angle (left/right)
R, T = look_at_view_transform(dist=2.7, elev=30, azim=0) 
cameras = PerspectiveCameras(device=device, R=R, T=T)

# --- Rasterizer Setup (Defines how to draw pixels) ---
raster_settings = RasterizationSettings(
    image_size=IMAGE_SIZE,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# --- Light Setup ---
# Place a point light
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

# --- Shader Setup (Defines how the mesh reacts to light) ---
# We will use a simple shader that uses vertex colors
# If your .obj has materials, this gets more complex.
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
for i, obj_file_path in enumerate(tqdm(obj_files, desc="Rendering Frames")):
    
    # 1. Load the OBJ file
    # 'load_obj' is a simple loader. It might not handle complex materials.
    verts, faces, aux = load_obj(obj_file_path, device=device)
    
    # 2. Create a Textures object
    # This assumes no texture map, just simple vertex colors (all white)
    # If your .obj has UVs (in aux.verts_uvs), you'd load a TexturesUV
    # For simplicity, we create a white color for all vertices
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb.to(device))

    # 3. Create a Meshes object
    # Note: Pytorch3D expects meshes in a batch.
    # We wrap our single mesh in a list.
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces.verts_idx.to(device)],
        textures=textures
    )

    # 4. Render the mesh
    # 'renderer' returns a batch of images. We take the first one [0].
    try:
        images = renderer(mesh)
        
        # 5. Convert and Save Image
        output_image = images[0, ..., :3].cpu().numpy()
        
        # Convert from 0-1 float to 0-255 uint8
        output_image_uint8 = (output_image * 255).astype('uint8')
        
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

# --- 7. (Optional) Cleanup ---
print("Cleaning up intermediate PNG files...")
for filename in image_filepaths:
    os.remove(filename)
os.rmdir(PNG_DIR)

print("Script finished.")
