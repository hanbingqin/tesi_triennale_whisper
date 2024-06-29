import speech_recognition as sr
import numpy as np
import whisper
import torch
import os
import time 
import keyboard
from queue import Queue

import csv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#Energy level for mic to detect.
energy_threshold = 970
#How real time the recording is in seconds.
record_timeout = 6
#How much empty space between recordings before we consider it a new line in the transcription.
phrase_timeout = 3
# The last time a recording was retrieved from the queue.
phrase_time = None
# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()

#setup per salvare i dati
csvfile =  open("generico.csv", mode= 'w')
fieldnames = ["testo_scritto", "tempo impiegato", "#parole/secondo" , "#lettere/secondo"]
#"=" used  as delimiter to be able to check for puntuation for larger models
writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='=')
writer.writeheader()
# We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
recorder = sr.Recognizer()
recorder.energy_threshold =energy_threshold
# Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
recorder.dynamic_energy_threshold = False

source = sr.Microphone(sample_rate=16000)
#carica modello 
#options = whisper.DecodingOptions( without_timestamps=True)
model =whisper.load_model("pp")

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


# Cue the user we are ready to start
print ("model loaded, press SPACE to start talking \n")
keyboard.wait("space")
# Create a background thread that will pass us raw audio bytes.
# We could do this manually but SpeechRecognizer provides a nice helper.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)


print(" \nTranscription:")
while True:
    try:
        now = time.time()
        # Pull raw recorded audio from the queue.
        if not data_queue.empty():

            delay_start = time.time()

            print ('trovata queue, elaboro...')
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > phrase_timeout:
                phrase_complete = True
           
            # Combine audio data from queue
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()

            # This is the last time we received new audio data from the queue.
            phrase_time = now
            
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
                #number of word and letter
                words = len(text.split())
                letters = len(text.replace(" ", ""))
                # delay transcription
                delay_end = time.time()
                delay_total= round(delay_end - delay_start, 3)
                #calcs for timing
                time_for_word = round(words/delay_total, 3)
                time_for_letter = round(letters/delay_total, 3)
                #write on csv
                writer.writerow({ "testo_scritto": text, "tempo impiegato": delay_total, "#parole/secondo" : time_for_word, "#lettere/secondo" : time_for_letter})
                transcription.append(text)

            elif len(text) == 0 :
                 delay_end = time.time()
            
            else:
                #number of word and letter
                words = len(text.split())
                letters = len (text.replace(" ", ""))
                # delay transcription
                delay_end = time.time()
                delay_total= round(delay_end - delay_start, 3)
                #calcs for timing
                time_for_word = round(words/delay_total, 3)
                time_for_letter = round(letters/delay_total, 3)
                #write on csv
                writer.writerow({ "testo_scritto": text, "tempo impiegato": delay_total, "#parole/secondo" : time_for_word, "#lettere/secondo" : time_for_letter})

                #write in line
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

    #for line in transcription:
    #    print(line)

print ('AAAAAAAAAAAA fuori dal ciclo fine di tutto AAAAAAAAA')
#mi ricavo il testo completo dalla composizione delle trascrizione di chunk singoli
#i obtain the transcription by composing the various chunk 
joined_string = " ".join(transcription)

writer.writerow({"testo_scritto": joined_string})

csvfile.close()
print(transcription)
print(joined_string)
#if __name__ == "__main__":
#    main()