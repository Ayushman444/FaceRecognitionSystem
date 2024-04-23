'''
We thank UMASS for the lABELLED FACES IN WILD DATASET which helped us a lot in training and tuning our model!!
the link for the  LFW site: https://vis-www.cs.umass.edu/lfw/

LFW serves as negatives while developer's own photos (by his consent) serve as positives
'''


import gdown 
def get_trainData():
    file_id = "1XGFx5lcXWRSRb5mkWvCTLO0I3U5DHEW9"
    output_file = "tripletData.zip"
    gdown.download(id=file_id, output=output_file)