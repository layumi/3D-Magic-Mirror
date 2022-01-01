file1 = open('template/sphere.obj', 'r')
Lines = file1.readlines()

with open('template/ellipsoid.obj', 'w') as fp:
    for line in Lines:
        info = line.split(' ')
        if info[0] == 'v':
            fp.write('v %f %f %f \n'%(float(info[1]), float(info[2])*2, float(info[3]) ))
        else:
            fp.write(line)
