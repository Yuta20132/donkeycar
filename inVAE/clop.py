from PIL import Image
import glob,os
from tqdm import tqdm

files = glob.glob(os.path.join('./data_folder/train_data/data*.jpg'))
print(files)

print(len(files))

output_dir='./data_mini/train'

for f in tqdm(files):
  try:
    image = Image.open(f)
    
  except OSError:
    print('Delete' + f)


  image_name=f.split('\\')[-1]
  image = image.resize((160,120))
  image.crop((0, 40, 160, 120)).save(f'{output_dir}/{image_name}', quality=95)

  