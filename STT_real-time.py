import speech_recognition as sr
import numpy as np
import whisper
import torch
import os
import time 
import keyboard
from queue import Queue

#Energy level for mic to detect.
#energy_threshold = 970
#How real time the recording is in seconds, avevo detto 5 qui e 2 sotto, in questo modo non farà mai in line decentemente
record_timeout = 5
#How much empty space between recordings before we consider it a new line in the transcription.
phrase_timeout = 3
# The last time a recording was retrieved from the queue.
phrase_time = None
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()

# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

source = sr.Microphone(sample_rate=16000)
#carica modello 
#options = whisper.DecodingOptions( without_timestamps=True)
model =whisper.load_model("medium")

transcription = ['']

with source:
    recorder.adjust_for_ambient_noise(source)
def record_callback(_, audio:sr.AudioData) -> None:
    """
    Threaded callback function to receive audio data when recordings finish.
    audio: An AudioData containing the recorded bytes.
    """
    # Grab the raw bytes and push it into the thread safe queue.
    data = audio.get_raw_data()
    data_queue.put(data)

# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

# Cue the user we are ready to start
print ("model loaded, start talking \n")

print("\nTranscription:")
while True:
    try:
        now = time.time()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():

            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > phrase_timeout:
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now
            
            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()
            
            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Read the transcription.
            result = model.transcribe(audio_np, language="it", fp16=torch.cuda.is_available())
            text = result['text'].strip()

            # If we detected a pause between recordings, add a new item to our transcription.
            # Otherwise edit the existing one.
            if phrase_complete:
                transcription.append(text)
                
            else:
                #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAELSEEEEEEEEEEEEEEEEEEEE")
                transcription[-1] = transcription[-1]+" "+text
                

            # Clear the console to reprint the updated transcription.
            os.system('cls')
            for line in transcription:
                print(line)
            # Flush stdout.
            print('', end='', flush=True)
        else:
            # Infinite loops are bad for processors, must sleep.
            time.sleep(0.5)
    except KeyboardInterrupt:
        data_queue.queue.clear()
        break

    if keyboard.is_pressed("space"):
        data_queue.queue.clear()
        print("Stopping recording, wait...")
        time.sleep(0.2)
        break
    #conditions to mute the mic
    if keyboard.is_pressed("m"):
        data_queue.queue.clear()
        print("muted...muted...")
        print("*press u to unmute*")
        stop_listening(wait_for_stop=False)
        time.sleep(0.5)
        while not keyboard.is_pressed("u"):
            time.sleep(0.2)
        stop_listening = recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
        print("unmuted, start talking")
        time.sleep(0.2)

print ('\n fuori dal ciclo, trascrizione finale: \n')
print(transcription)
