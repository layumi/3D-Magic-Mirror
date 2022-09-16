import os

root = '../log/'
dir_list = []

for f in os.listdir(root):
    dir_name = root+f
    model_path = dir_name+'/ckpts/latest_ckpt.pth'
    if os.path.isfile(model_path):
        print(model_path)
        os.system('rm %s'%model_path)
