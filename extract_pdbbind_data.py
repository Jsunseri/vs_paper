import sys,math
# pass in the index file and desired outname

exp = {}
exp['mM'] = -3
exp['uM'] = -6
exp['nM'] = -9
exp['pM'] = -12
exp['fM'] = -15
exp['aM'] = -18
exp['zM'] = -21
exp['yM'] = -24

outfile = open(sys.argv[2], 'w')
with open(sys.argv[1], 'r', errors='ignore') as f:
    for line in f:
        if line.startswith('#'):
            continue
        contents = line.split()
        name = contents[0]
        Kd = contents[3].split('=')[-1]
        num = float(''.join([char for char in Kd if char.isdigit() or char == '.']))
        unit = ''.join([char for char in Kd if char.isalpha()])
        expval = exp[unit]
        pKd = -(math.log10(num * 10**(expval)))
        outfile.write('%s %s%s %0.4f\n' %(name, num, unit, pKd))
outfile.close()
