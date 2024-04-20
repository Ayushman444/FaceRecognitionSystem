'''
This is just a slice of BLFW dataset that we extracted to test how much better our fine tuned model has performed 
on an India Specific Dataset compared to original pre-trained model
'''
def get_IBLFW():
    import gdown
    file_id = '1QvOlcOqQCeq1r3fegHpE9fnps3h-Aij4'
    output_file = 'indianData.zip'
    print("Be sure to be running the program in main project directory!!")
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output_file, quiet=False)
    print('Success!')