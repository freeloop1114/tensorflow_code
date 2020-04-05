
import sys

s = '1\t1,2,3,4\t5,6,7,8\t0,0,0:1,1,1:2,2,2'
with open(sys.argv[1], 'w') as f:
    f.write('%s\n' % (s))
