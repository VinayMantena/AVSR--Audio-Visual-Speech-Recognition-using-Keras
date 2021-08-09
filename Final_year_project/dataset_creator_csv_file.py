import os
import numpy as np
import pandas as pd
import imutils
from imutils import face_utils
import dlib
import cv2
from scipy.fftpack import fft
from scipy import signal
from scipy.io import wavfile
import librosa
#from DEFAULTS import SHAPE_PREDICTOR_LOCATION,CSV_FILE,TRAIN_DATA_LOCATION
TRAIN_DATA_LOCATION = ""#trainpath_to_the_videodataset_be_inserted
train_audio_path = ""#trainpath_to_the_audiodataset_be_inserted
SHAPE_PREDICTOR_LOCATION = "./shape_predictor_68_face_landmarks.dat"
CSV_FILE = "./avsr_test_data_eng_same_test_size_train_v2.csv"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_LOCATION)

X  = []
Y  = []
label_dictionary = dict()


def init():
    label_vector = [each_word for each_word in os.listdir(TRAIN_DATA_LOCATION) ]
    for (x,y) in enumerate(label_vector):
        label_dictionary.update({y:x})

def process_Video(video):
    print("================================")
    X = []
    Y = []
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS : " + str(fps))
    if(not cap.isOpened()):
        print(" .....Cannot open video stream or file!.....")
    print(" video/camera opened...")
    print("********************************************")
    while(cap.isOpened()):
        frameId = cap.get(1)
        try:
            ret, image = cap.read()
            if(ret==False):
                break
            else:
                print("Frame ID :"+ str(frameId))
                #image = imutils.rotate_bound(image,angle=90)
                #image = imutils.resize(image,width = 500)
                gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                #cv2.imshow("gray",image)
                faces = face_detector(gray_image,1)
                for (i,rect) in enumerate(faces):
                    shape = shape_predictor(gray_image,rect)
                    shape = face_utils.shape_to_np(shape)

                    for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        if(name=='mouth'):
                            #cpyImage = image.copy()
                            X_frame = []
                            Y_frame = []
                            for (x,y) in   shape[i:j]:
                                X_frame.append(x)
                                Y_frame.append(y)
                    X.append(X_frame)
                    Y.append(Y_frame)
                            # (x,y,w,h) = cv2.boundingRect(np.array(shape[i:j]))
                            # roi = imutils.resize(cpyImage[y:y+h,x:x+w],width=250,inter=cv2.INTER_CUBIC)
                            # cv2.imshow("Image",roi)

                    print("================================================")
            # if(cv2.waitKey(1) & 0XFF == ord('q')):
            #                             print("........... Stopping..........")

            #                             break;
        except Exception as e:
            print("ERROR OCCURED......." + e)
            break;
    cap.release()
    cv2.destroyAllWindows()
    return X,Y

def process_Arrays(X,Y):
    X_sd_array = []
    Y_sd_array = []
    X_sd_array = np.std(X,axis = 0).astype('float32')
    Y_sd_array = np.std(Y,axis = 0).astype('float32')
    return X_sd_array,Y_sd_array


def process(training_Dataframe):
    temp_dictionary = {}

    for each_word in os.listdir(TRAIN_DATA_LOCATION):
        i=0
        print("#########" + each_word + "##########")
        word_folder = os.path.join(TRAIN_DATA_LOCATION,each_word)

        print("Current Directory  :" + word_folder )

        for each_video in os.listdir(word_folder):

            i+=1

            print("Processing video " + each_video)
            print("Video count " + str(i))
            # samples, sample_rate = librosa.load(train_audio_path + '/' + each_word + '/' + each_video[:-3] + "wav", sr = 16000)
            # samples = librosa.resample(samples, sample_rate, 8000)
            #
            # if(len(samples) != 8363):
            #     continue
            X, sample_rate = librosa.load(train_audio_path + '/' + each_word + '/' + each_video[:-3] + "wav", sr = 16000)
            stft = np.abs(librosa.stft(X))
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
            feature_array = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            feature_vector_audio  = np.around(feature_array,decimals=3)
            X,Y = process_Video(os.path.join(word_folder,each_video))
            X,Y = process_Arrays(X,Y)
            feature_vector = np.r_[X,Y]
            feature_vector  = np.around(feature_vector,decimals=3)
            feature_vector = np.concatenate((feature_vector,feature_vector_audio), axis = 0)
            #print(feature_vector)


            temp_dictionary = dict(enumerate(feature_vector))
            temp_dictionary.update({'label':label_dictionary[each_word]})

            training_Dataframe = training_Dataframe.append(temp_dictionary, ignore_index=True)
            training_Dataframe = training_Dataframe.astype({'label':'int32'})
            print(training_Dataframe)
    return training_Dataframe


if __name__ == "__main__":

    init()
    training_Dataframe = pd.DataFrame(dtype='float32')
    training_Dataframe = process(training_Dataframe)
    training_Dataframe = training_Dataframe.sample(frac=1).reset_index(drop=True)
    print(training_Dataframe)
    training_Dataframe.to_csv(CSV_FILE,index=False)
    print("Done.....")
