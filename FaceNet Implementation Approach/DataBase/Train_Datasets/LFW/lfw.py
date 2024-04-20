'''
We thank UMASS for the lABELLED FACES IN WILD DATASET which helped us a lot in training and tuning our model!!
the link for the  LFW site: https://vis-www.cs.umass.edu/lfw/
'''

def get_lfw():        
    import gdown
    file_id = '1AOvPmGsReR1hG-_T-gxxESEiMqxIFQ5M'
    print("Be sure to be running the program in main project directory!!")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', quiet=False)
    print('Success!!')