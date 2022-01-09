file1 = open('template/sphere2.obj', 'r')
Lines = file1.readlines()

with open('template/ellipsoid2.obj', 'w') as fp:
    for line in Lines:
        info = line.split(' ')
        if info[0] == 'v':
            fp.write('v %f %f %f \n'%(float(info[1]), float(info[2])*2, float(info[3]) ))
        else:
            fp.write(line)
