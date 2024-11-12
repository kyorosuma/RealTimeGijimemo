import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np

# 録音設定
SAMPLE_RATE = 44100  # サンプルレート
DURATION = 5  # 録音時間（秒）

def record_audio(filename="output.wav"):
    print("録音開始...")
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=2)
    sd.wait()  # 録音終了まで待機
    print("録音終了")
    write(filename, SAMPLE_RATE, audio_data)  # WAVファイルとして保存

def transcribe_audio(filename="output.wav", output_file="transcription.txt"):
    model = whisper.load_model("medium")
    result = model.transcribe(filename)
    transcription = result["text"]
    print("文字起こし結果:", transcription)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"文字起こし結果が'{output_file}'に保存されました。")
    except Exception as e:
        print(f"ファイルの書き込みに失敗しました: {e}")


# 録音と文字起こしの実行
record_audio("output.wav")
transcribe_audio("output.wav", "transcription.txt")
