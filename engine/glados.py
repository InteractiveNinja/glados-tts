import os
import re
import time
from subprocess import call

import torch
from scipy.io.wavfile import write
from utils.tools import prepare_text


class GladosEngine:
    def __init__(self, play_file=False, cache=True):
        self.play_file = play_file
        self.cache = cache
        self._prepare_engine()

    def _prepare_engine(self):
        # Check if env are set
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
        os.environ['PHONEMIZER_ESPEAK_PATH'] = 'C:\Program Files\eSpeak NG\espeak-ng.exe'

        print("Initializing TTS Engine...")

        # Select the device
        if torch.is_vulkan_available():
            self.device = 'vulkan'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        model_folder = os.path.join(os.getcwd(), "models")

        # Load models
        self.glados = torch.jit.load(os.path.join(model_folder, "glados.pt"))
        self.vocoder = torch.jit.load(os.path.join(model_folder, 'vocoder-gpu.pt'), map_location=self.device)

    def glados_tts(self, text):
        audio_folder_path = os.path.join(os.getcwd(), "cache")

        # ensure folder exists
        if not os.path.exists(audio_folder_path):
            os.mkdir(audio_folder_path)

        audio_filename = "GLaDOS-tts-" + text.replace(" ", "-").lower() if self.cache else "GLaDOS-tts-output"
        audio_filename = re.sub(r"[$&+,:;=?@#|'<>.^*()%!]", "", audio_filename) + ".wav"

        audio_file_path = os.path.join(audio_folder_path, audio_filename)

        if self.cache:
            if not os.path.exists(audio_file_path):
                self._generate_file(text, audio_file_path)
        else:
            self._generate_file(text, audio_file_path)

        if self.play_file:
            self._play_audio(audio_file_path)

        return audio_file_path

    def _generate_file(self, text, file_path):
        # Tokenize, clean and phonemize input text
        x = prepare_text(text).to('cpu')
        with torch.no_grad():
            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados.generate_jit(x)

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            print("\033[1;94mINFO:\033[;97m The audio sample took " + str(
                round((time.time() - old_time) * 1000)) + " ms to generate.")

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')


            # Write audio file to disk
            # 22,05 kHz sample rate
            write(file_path, 22050, audio)

    def _play_audio(self, filepath):
        try:
            import winsound
            winsound.PlaySound(filepath, winsound.SND_FILENAME)
        except ImportError:
            try:
                call(["aplay", filepath])
            except FileNotFoundError:
                call(["pw-play", filepath])
