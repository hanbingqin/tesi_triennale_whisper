import speech_recognition as sr
import numpy as np
import whisper
import torch
import os
import time 
from queue import Queue


#Dimensione chunk audio
record_timeout = 3
#Quantità di silenzio tra le registrazioni prima di iniziare una nuova riga nella trascrizione.
phrase_timeout = 1
#L'ultima volta in cui una registrazione è stata recuperata dalla queue.
phrase_time = None
#Coda thread-safe per il passaggio di dati con record_callback.
data_queue = Queue()

#Utilizziamo SpeechRecognizer per registrare l'audio perché offre una  funzionalità interessante che consente di rilevare la fine del parlato.
recorder = sr.Recognizer()
#La compensazione dinamica dell'energia abbassa drasticamente la soglia di energia al punto da rendere recorder sempre attiva.
recorder.dynamic_energy_threshold = False

source = sr.Microphone(sample_rate=16000)
#Carica il modello Whisper di dimensione "medium".
model =whisper.load_model("tiny", device="cpu")
transcription = ['']

with source:
    # print("step1:adjust for ambient noise")
    recorder.adjust_for_ambient_noise(source)
def record_callback(_, audio:sr.AudioData) -> None:
    #Estrae i byte dall'oggetto audio.
    # print("step2:start to get raw data")
    data = audio.get_raw_data()
    #Inserisce i dati nella coda thread-safe.
    data_queue.put(data)

#Crea un thread in background che gestirà la registrazione e invierà i dati grezzi alla coda.
recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

#Avvisa l'utente che il setup è terminato
print ("model loaded, start talking \n")

print("\n*press m to mute*\nTranscription:")
while True:
    try:
        now = time.time()
        #Recupera l'audio grezzo registrato dalla coda.
        if not data_queue.empty():

            phrase_complete = False
            #Se è trascorso abbastanza tempo tra le registrazioni,considera la frase completata.
            if phrase_time and now - phrase_time > phrase_timeout:
                phrase_complete = True
            #Salviamo il tempo di arrivo della registrazione corrente.
            phrase_time = now
            
            #Combina i dati audio dalla coda con byte join.
            # print("step3: get audio data")
            audio_data = b''.join(data_queue.queue)
            data_queue.queue.clear()
            
            #Converte i dati da interi a 16 bit in virgola mobile a 32 bit
            #Normalizza considerando una PCM di 32768 Hz massimo.
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            #Effettua e salva la trascrizione.
            result = model.transcribe(audio_np, language="zh", fp16=torch.cuda.is_available(), initial_prompt="以下是普通话的句子")
            text = result['text'].strip()
            print("测试结果：",text)
            #Aggiungi una nuova riga alla trascrizione in caso di pausa sufficiente.
            if phrase_complete:
                transcription.append(text)
            #Altrimenti modifica quella esistente.     
            else:
                transcription[-1] = transcription[-1]+" "+text
                

            #Cancella la console per stampare la trascrizione aggiornata.
            # # os.system('cls')
            # for line in transcription:
            #     print("测试结果：",line)
            #Svuota lo stdout per evitare righe sovrapposte.
            # print('', end='', flush=True)
        else:
            #Per evitare cicli infiniti.
            time.sleep(0.5)
    #Gestione delle interruzioni da tastiera (Ctrl+C).
    except KeyboardInterrupt:
        data_queue.queue.clear()
        break
    #Gestione terminazione programma premendo spazio.

#Una volta terminato il programma, stampiamo l'intera trascrizione.
print ('\n fuori dal ciclo, trascrizione finale: \n')
print(transcription)
