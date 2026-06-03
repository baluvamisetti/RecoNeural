content = open('train.py').read()
content = content.replace('optimizer, mode="max", factor=0.5, patience=2, verbose=True', 'optimizer, mode="max", factor=0.5, patience=2')
open('train.py', 'w').write(content)
print('Fixed!')
