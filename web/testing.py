import sys

print('Output from python')
print('Data[1]: {}'.format(sys.argv[1]))
print('Data type: {}'.format(type(sys.argv[1])))

input = sys.argv[1].split('\n')
for d, i in enumerate(input):
    print('{} -> {}'.format(str(d), str(i)))
