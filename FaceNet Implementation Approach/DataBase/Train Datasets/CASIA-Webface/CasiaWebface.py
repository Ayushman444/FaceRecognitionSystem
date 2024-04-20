'''
We Thank CASIA-Webface for providing such an extensive set of images for face recognition and model training
We couldn't utilize this dataset for training due to low compute availability at our end but the size and ability 
of team CASIA to compress the dataset to a few GBs using .rec file type is amazing

More About CASIA-WEBFACE : https://paperswithcode.com/dataset/casia-webface
'''

import gdown
file_id = '1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l'
gdown.download(f'https://drive.google.com/uc?id={file_id}', quiet=False)