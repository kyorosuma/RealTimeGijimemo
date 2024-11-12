from flask import Flask, render_template, jsonify, stream_with_context, Response
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import threading
import time
import numpy as np
import queue

app = Flask(__name__)
is_recording = False
transcriptions = []
SAMPLE_RATE = 44100
DURATION = 5
audio_queue = queue.Queue()

def record_audio():
    """録音を5秒ごとに継続して行い、音声データをキューに追加"""
    global is_recording
    while is_recording:
        filename = f"output_{int(time.time())}.wav"
        audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=2)
        sd.wait()
        write(filename, SAMPLE_RATE, np.int16(audio_data * 32767))
        audio_queue.put((filename, time.time()))  # キューに音声ファイル名と録音時刻を追加

def transcribe_audio():
    """キュー内の音声ファイルを順番に文字起こし"""
    model = whisper.load_model("small")
    global is_recording, transcriptions
    while is_recording or not audio_queue.empty():
        if not audio_queue.empty():
            filename, start_time = audio_queue.get()
            result = model.transcribe(filename)
            end_time = start_time + DURATION
            transcriptions.append(f"{start_time:.0f}-{end_time:.0f}[s] {result['text']}")
            audio_queue.task_done()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_recording():
    global is_recording
    is_recording = True
    threading.Thread(target=record_audio).start()      # 録音スレッドを開始
    threading.Thread(target=transcribe_audio).start()  # 文字起こしスレッドを開始
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False  # 録音を停止し、キュー内の処理を続行
    return jsonify({"status": "stopped"})

@app.route('/transcriptions')
def stream_transcriptions():
    def generate():
        global transcriptions
        while is_recording or transcriptions or not audio_queue.empty():
            if transcriptions:
                yield f"data: {transcriptions.pop(0)}\n\n"
            time.sleep(1)
        yield "event: end\ndata: 終了\n\n"  # 文字起こし完了後にendイベントを送信

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True)
