import os
import librosa
import math
import json

DATASET_PATH ="Path_for_GTZAN_Dataset"
JSON_PATH ="data.json"

SAMPLE_RATE = 22050 #defaut Librosa sample rate

SAMPLES_PER_TRACK = SAMPLE_RATE * 30  # samples per track (each track has 30s)


def save_mfcc(dataset_path,json_path,n_mfcc=13,n_fft=2048,hop_length=512,num_segments=10):

    # store the data as a dictionary
    data = {
        "mapping":[],
        "mfcc":[],
        "labels":[]
    }

    num_sample_per_seg = int(SAMPLES_PER_TRACK / num_segments) # sample in each segmenet in our case we have 10 segments
    
    expected_num_mfcc_vectors_per_seg = math.ceil(num_sample_per_seg / hop_length) # expected length or the mffcs
    
    
    # go throught all the genre
    for i,(dirpath,dirname,filesnames) in enumerate(os.walk(dataset_path)):

        #ensure we're not at the root level
        if dirpath is not dataset_path:

            #save the labels
            dirpath_comp = dirpath.split("\\") #genre/blues => ["genre","blues]
            semantic_laber = dirpath_comp[-1] # we take the last value which will be the genre ["blues"]
            data["mapping"].append(semantic_laber) # store the genre in the dictionary 
            print("\nProcessing {}".format(semantic_laber))

            #process files for a genre
            for f in filesnames:

                #load audio
                file_path= os.path.join(dirpath,f)
                signal,sr = librosa.load(file_path,sr=SAMPLE_RATE)

                #process segs extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_sample_per_seg * s # calculate the starting sampe
                    finish_sample = start_sample + num_sample_per_seg # calculate the finish sample


                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample], #extraction mfccs
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store mfcc if it has the expected lenght
                    if len(mfcc) == expected_num_mfcc_vectors_per_seg:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print('{}, segment: {}'.format(file_path,s+1))

    with open(json_path,'w') as fp:
        json.dump(data,fp,indent=4)


save_mfcc(DATASET_PATH,JSON_PATH,num_segments=10)
