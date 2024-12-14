import os
import time
import whisper
import OpenAI
from pathlib import Path
from pydub import AudioSegment
#from IPython.display import Audio


def chunk_audio(input_file, temp_dir, chunk_size_mb=20):
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    audio = AudioSegment.from_file(input_file)

    audio_size = os.path.getsize(input_file)
    chunk_duration_ms = len(audio) * (chunk_size_bytes / audio_size)

    os.makedirs(temp_dir, exist_ok=True)

    for i in range(0, len(audio), int(chunk_duration_ms)):
        chunk = audio[i:i + int(chunk_duration_ms)]
        chunk_file = os.path.join(temp_dir, f"chunk_{i // int(chunk_duration_ms)}.mp3")
        chunk.export(chunk_file, format="mp3")
        yield chunk_file


def milliseconds_until_sound(sound, silence_threshold_in_decibels=-20.0, chunk_size=10):
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold_in_decibels and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def trim_start(filepath):
    path = Path(filepath)
    directory = path.parent
    filename = path.name
    audio = AudioSegment.from_file(filepath, format="wav")
    start_trim = milliseconds_until_sound(audio)
    trimmed = audio[start_trim:]
    new_filename = directory / f"trimmed_{filename}"
    trimmed.export(new_filename, format="wav")
    return trimmed, new_filename


def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i)<128)


def transcribe_audio(file,output_dir):

    # with open("prompts/notes.txt", "r") as file:
    #     prompt_template = file.read()
    
    prompt = (
        f"Questo è il frammento di una lezione universitaria di teologia. L'argomento "
        f"è l'introduzione alla sacra scrittura, nella lezione sono raccontati personaggi biblici dell'antico testamento come:"
        f"Samuele, Saul, Rut, Davide, Salomone, Roboamo, Geroboamo."
    )

    audio_path = os.path.join(output_dir, file)
    with open(audio_path, 'rb') as audio_data:
        transcription = model.transcribe(model="large", 
                                           file=audio_data, 
                                           language='it', 
                                           verbose=True,  
                                           initial_prompt=prompt)
        return transcription.text
    

def punctuation_assistant(ascii_transcript):

    system_prompt = """Sei un assistente che aiuta ad aggiungere la punteggiatura al testo.
    Preserva le parale originali e aggiungi solo la punteggiatura necessaria per produrre un discorso scorrevole. 
    Dovresti aggiungere quindi, virgole, punti alla fine dei periodi, punti di domanda, lettere maiusole e 
    formattazione quando necessaria.
    Utilizza solo il contesto fornito. Se non c'è il contesto necessario dì soltanto: 'Nessun contesto fornito' \n
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response


def biblical_assistant(ascii_transcript):
    system_prompt = """Sei un dotto assistente universitario esperto di teologia e antico testamente.
    Ti sarà fornita la trascrizione di una lezione universitaria di teologia sull'introduzione alla sacra scrittura, la
    lezione riguarderà principalmete fatti biblici e punterà a trasmettere la giusta chiave di lettura per comprenderli,
    quindi contestualizzando i fatti. Il tuo compito è correggere eventuali nomi dell'antico testamento che dovessero essere stati trascritti male,
    completare eventuali discorsi incompleti, approfondendo l'argomento con quello che sai per produrre un testo scorrevole e comprensibile sul quale 
    poi permettere agli studenti di studiare. Cerca non non tralasciare nulla, i dettagli e aneddoti storici possono essere importanti per eccellere nell'esame.
    Inzia ad eleborare il testo con una piccola introduzione della lezione trattata dicendo: 'Buongiorno, la lezione di oggi riguarderà... aggiungi qui l'introduzione...' """
    response = client.chat.completions.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": ascii_transcript
            }
        ]
    )
    return response


if __name__ == '__main__':

    input_dir = "media/input/2_Intro_Sacrascrittura_Grassilli.WAV"
    output_dir = "media/output/2_Intro_Sacrascrittura_Grassilli_trascritta.txt"
    temp_dir = "media/temp"

    # Initialize Whisper model and assistant
    print("Inizialising the models")
    client = OpenAI()
    model = whisper.load_model("large")
    print("\n")

    # Trim the start of the original audio file to remove silence from audio
    print("Trimming the audio file")
    trimmed_audio, trimmed_filename = trim_start(input_dir)
    trimmed_audio = AudioSegment.from_wav(trimmed_filename)
    print("\n")

    # here the aduio file is divided in chunks
    one_minute = 1 * 60 * 1000  # Duration for each segment (in milliseconds)

    start_time = 0  # Start time for the first segment

    i = 0  # Index for naming the segmented files

    if not os.path.isdir(temp_dir):  # Create the output directory if it does not exist
        os.makedirs(temp_dir)

    while start_time < len(trimmed_audio):  # Loop over the trimmed audio file
        segment = trimmed_audio[start_time:start_time + one_minute]  # Extract a segment
        segment.export(os.path.join(temp_dir, f"trimmed_{i:02d}.wav"), format="wav")  # Save the segment
        start_time += one_minute  # Update the start time for the next segment
        i += 1  # Increment the index for naming the next file

    # Get list of trimmed and segmented audio files and sort them numerically
    audio_files = sorted(
        (f for f in os.listdir(trimmed_audio) if f.endswith(".wav")),
        key=lambda f: int(''.join(filter(str.isdigit, f)))
    )

    print("Starting transcription")
    transcription_start = time.time()
    transcriptions = [transcribe_audio(file, temp_dir) for file in audio_files]
    full_transcript = ' '.join(transcriptions)
    print(f"Transciption made in: {time.time()-transcription_start} seconds")
    print("\n")
    print(full_transcript[0:1000])

    # Remove non-ascii characters from the transcript
    print("Removing non ascii characters...")
    ascii_transcript = remove_non_ascii(full_transcript)
    print("\n")

    # Use punctuation assistant function
    print("Adding punctuation...")
    response = punctuation_assistant(ascii_transcript)
    punctuated_transcript = response.choices[0].message.content
    print("\n")

    print("Invoking biblical assistant...")
    response = biblical_assistant(punctuated_transcript)
    final_transcript = response.choices[0].message.content
    print("\n")

    print("final transcription:")
    print(final_transcript["text"][0:1000])

    with open(output_dir, "a", encoding="utf-8") as f:
        f.write(final_transcript["text"])
    print("trascription saved. Ending job")