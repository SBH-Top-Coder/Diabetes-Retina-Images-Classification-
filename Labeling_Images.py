import os 
from pathlib import Path
import pandas as pd 
import shutil
F_Images_Path = '/home/semi/Image_Classification/Full_Images'
if not os.path.exists(F_Images_Path):
        os.makedirs(F_Images_Path)

#  Preparing Lines : 
Lines = ['image,level']
nb = 0 
for path in Path('/home/semi/Image_Classification/Images/clas0').rglob('*'):
    path_to_go = F_Images_Path+ f'/image{nb}.jpeg'
    shutil.copyfile(path,path_to_go)
    Lines.append(f'image{nb}'+ ',0')
    nb = nb + 1 

    
for path in Path('/home/semi/Image_Classification/Images/clas1').rglob('*'):
    path_to_go = F_Images_Path+ f'/image{nb}.jpeg'
    shutil.copyfile(path,path_to_go)
    Lines.append(f'image{nb}'+ ',1')
    nb = nb + 1 

    
for path in Path('/home/semi/Image_Classification/Images/clas2').rglob('*'):
    path_to_go = F_Images_Path+ f'/image{nb}.jpeg'
    shutil.copyfile(path,path_to_go)
    Lines.append(f'image{nb}'+ ',2')
    nb = nb + 1 

    
for path in Path('/home/semi/Image_Classification/Images/clas3').rglob('*'):
    path_to_go = F_Images_Path+ f'/image{nb}.jpeg'
    shutil.copyfile(path,path_to_go)
    Lines.append(f'image{nb}'+ ',3')
    nb = nb + 1 

    
# Preparing Labels : 
with open('/home/semi/Image_Classification/Labels.csv', 'w') as f:
    for line in Lines:
        f.write(line)
        f.write('\n')
