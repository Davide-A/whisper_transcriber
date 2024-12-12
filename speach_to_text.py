import os
import time
import whisper
from pydub import AudioSegment


def chunk_audio(input_file, temp_dir, chunk_size_mb=25):
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    print("here")
    audio = AudioSegment.from_file(input_file)

    audio_size = os.path.getsize(input_file)
    chunk_duration_ms = len(audio) * (chunk_size_bytes / audio_size)

    os.makedirs(temp_dir, exist_ok=True)

    for i in range(0, len(audio), int(chunk_duration_ms)):
        chunk = audio[i:i + int(chunk_duration_ms)]
        chunk_file = os.path.join(temp_dir, f"chunk_{i // int(chunk_duration_ms)}.mp3")
        chunk.export(chunk_file, format="mp3")
        yield chunk_file

if __name__ == '__main__':
    input_dir = "media/input"
    output_dir = "media/output"
    temp_dir = "media/temp"

    # Initialize Whisper model
    model = whisper.load_model("small")

    prompt = (
        f"Questo è il frammento di una lezione universitaria di teologia. L'argomento "
        f"è l'introduzione alla sacra scrittura, la lezione è tenuta da un professore."
    )

    # Process each audio file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".WAV"):
            input_path = os.path.join(input_dir, file_name)
            print(f"Processing: {input_path}")

            start = time.time()

            # Split the audio into chunks and transcribe each chunk
            for chunk_file in chunk_audio(input_path, temp_dir):
                print(f"Transcribing chunk: {chunk_file}")

                # Transcribe with Whisper
                result = model.transcribe(chunk_file, language='it', verbose=True, initial_prompt=prompt)
                output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_transcription.txt")

                # Save transcription
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(result["text"] + "\n")

                print(f"Processed chunk {chunk_file} in {time.time() - start:.2f} seconds")

            print(f"Finished processing {input_path} in {time.time() - start:.2f} seconds")

    print("Job finished")