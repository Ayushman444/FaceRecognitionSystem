'''
We thank IEEE for providing us with the balanced labelled face in wild dataset for free use for research and educational purposes
The link for same can be found here: https://ieee-dataport.org/documents/balanced-faces-wild

This dataset was used by us to test our model and calculate its accuracy
'''

def get_BLFW():
    import gdown
    file_id = '1H7kDTt4fkdmPLJBQtXTVBI0rJ6IB1bXY'
    output_file = 'datasetBLFW.zip'
    print("Be sure to be running the program in main project directory!!")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)
    print('Success!')