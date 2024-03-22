import os
import io
import time
import ffmpeg
import numpy as np
from base64 import b64decode
from pydub import AudioSegment
from transformers import pipeline
from IPython.display import HTML, Audio
from google.colab.output import eval_js
from scipy.io.wavfile import read as wav_read

AUDIO_HTML = """
<script>
var my_div = document.createElement("DIV");
var my_p = document.createElement("P");
var my_btn = document.createElement("BUTTON");
var t = document.createTextNode("Press to start recording");

my_btn.appendChild(t);
my_div.appendChild(my_btn);
document.body.appendChild(my_div);

var base64data = 0;
var reader;
var recorder, gumStream;
var recordButton = my_btn;

var handleSuccess = function(stream) {
  
  gumStream = stream;
  var options = {  mimeType : 'audio/webm;codecs=opus' };         

  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = function(e) {            
      var url = URL.createObjectURL(e.data);
      var preview = document.createElement('audio');
      preview.controls = true;
      preview.src = url;
      document.body.appendChild(preview);

      reader = new FileReader();
      reader.readAsDataURL(e.data); 
      reader.onloadend = function() { base64data = reader.result; }
  };
  
  recorder.start();
  };

recordButton.innerText = "Voice Recording Started";

navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);

function toggleRecording() {
  if (recorder && recorder.state == "recording") 
  {
      recorder.stop();
      gumStream.getAudioTracks()[0].stop();
      recordButton.innerText = "Voice Recording Ended"
  }
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }

var data = new Promise(resolve=>{

    recordButton.onclick = () => {
      toggleRecording()
      sleep(2000).then(() => { 
                      resolve(base64data.toString()) 
                      } );
    }
});
      
</script>
"""

def get_audio():
  
  display(HTML(AUDIO_HTML))
  data = eval_js("data")
  binary = b64decode(data.split(',')[1])
  
  process = (ffmpeg
    .input('pipe:0')
    .output('pipe:1', format='wav')
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True, overwrite_output=True)
  )
  output, err = process.communicate(input=binary)
  
  riff_chunk_size = len(output) - 8
  # Break up the chunk size into four bytes, held in b.
  q = riff_chunk_size
  b = []
  for i in range(4):
      q, r = divmod(q, 256)
      b.append(r)

  # Replace bytes 4:8 in proc.stdout with the actual size of the RIFF chunk.
  riff = output[:4] + bytes(b) + output[8:]

  sr, audio = wav_read(io.BytesIO(riff))

  return audio, sr

def save_audio(audio, sr, atype='mp3', name='audio'):

  fname = name + '.' + atype
  if atype == 'mp3':
      audio_segment = AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=audio.dtype.itemsize, channels=1)
      audio_segment.export(fname, format="mp3")
  elif atype == 'wav':
      wav_write(fname, sr, audio)
  else:
      print(f"ERROR: {atype} not supported!")

def stt_model(name='openai/whisper-small'):
   
   model = pipeline('automatic-speech-recognition', 
                      model = name, 
                      device = 0)
   return model

def transcribe(model, file_name="audio.mp3", timeout=30, sleep_interval=5):

    start_time = time.time()

    while not os.path.exists(file_name):
        if time.time() - start_time > timeout:
            return None

        time.sleep(sleep_interval)

    text = model(file_name)
    return text['text']