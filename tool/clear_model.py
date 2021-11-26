import os

root = '../log/'

for f in os.listdir(root):
    dir_name = root+f
    for ff in os.listdir(dir_name):
        if 'epoch' in ff:
            epoch = int(ff.split('_')[1])
            if (not epoch%50 == 0) or epoch==0:
                print(dir_name+'/'+ff)
                os.remove(dir_name+'/'+ff)
