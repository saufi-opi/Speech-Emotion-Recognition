import pyaudio
import wave
import tensorflow as tf
import numpy as np
import librosa
import joblib

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050

p = pyaudio.PyAudio()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model2.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the scaler from the file
scaler = joblib.load('scaler.pkl')

def start_record(seconds):
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHANNELS)

    #start record
    frames= []

    for i in range(0, int(RATE/CHUNK*seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    #finish record
    stream.stop_stream()
    stream.close()
    return frames

def get_input(file_path):
    data,sr=librosa.load(file_path, sr=RATE)
    data=librosa.effects.trim(data, top_db = 30)[0]
    mfcc=np.mean(librosa.feature.mfcc(y=data, sr=RATE, n_mfcc=50).T, axis=0)
    mfcc=np.expand_dims(mfcc,0)
    mfcc=scaler.transform(mfcc)
    mfcc=np.expand_dims(mfcc,-1)
    return mfcc

def predict(input_tensor):
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    class_predicted=np.argmax(output_data)
    return class_predicted,output_data

def start():
    while True:

        frames=start_record(3)

        w = wave.open("output.wav", 'wb')
        w.setnchannels(CHANNELS)
        w.setsampwidth(p.get_sample_size(FORMAT))
        w.setframerate(RATE)
        w.writeframes(b''.join(frames))
        w.close()

        mfcc=get_input("output.wav")
        predicted,prob=predict(mfcc)
        emotions=['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        print(f"Class predicted: Class {emotions[predicted]}")

start()


