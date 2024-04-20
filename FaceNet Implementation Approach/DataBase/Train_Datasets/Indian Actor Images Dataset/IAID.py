'''
We thank Saurav Banerjee for compiling together a list of great datasets on kaggle
https://www.kaggle.com/datasets/iamsouravbanerjee/indian-actor-images-dataset

For providing the dataset 

We Used This Dataset for fine tuning our pre-trained model setup for indian audience
'''
def get_IAID():
    import gdown
    file_id = '1DFNBzmlQpUvPsIG-SUThRpABk2v8ulyY'
    print("Be sure to be running the program in main project directory!!")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', quiet=False)
    print('Success!!')