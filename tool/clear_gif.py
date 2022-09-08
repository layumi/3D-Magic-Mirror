import os

root = '../log/'
dir_list = []

for f in os.listdir(root):
    dir_name = root+f
    model_path = dir_name+'/ckpts/latest_ckpt.pth'
    if not os.path.isfile(model_path):
        print(dir_name) # remove empty folders
        dir_list.append(dir_name)
    for ff in os.listdir(dir_name):
        if 'epoch' in ff:
            epoch = int(ff.split('_')[1])
            if (not epoch%20 == 0) or epoch<100:
                print(dir_name+'/'+ff)
                os.remove(dir_name+'/'+ff)


#for dir_name in dir_list:
#    os.system('rm -r %s'%dir_name)
