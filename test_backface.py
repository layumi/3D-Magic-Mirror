import kaolin as kal
from smr_utils import face_clocks
import torch
mesh_name = './template/ellipsoid.obj'
mesh = kal.io.obj.import_mesh(mesh_name, with_materials=True)

clocks = face_clocks(mesh.vertices.unsqueeze(0), mesh.faces)

print(torch.sum(clocks>0))
print(torch.sum(clocks<0))

