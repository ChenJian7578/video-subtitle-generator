# -*- coding: UTF-8 -*-
"""
@author: Fang Yao
@file  : main.py
@time  : 2022/04/27 22:55
@desc  : 主入口文件
"""
import io
import multiprocessing
import subprocess
import tempfile
import warnings
warnings.filterwarnings('ignore')
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from utils.formatter import FORMATTERS
import librosa
import os
import audioop
import torch
import config
import wave
import math


class AudioRecogniser:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(config.ASR_MODEL_PATH)
        self.model = Wav2Vec2ForCTC.from_pretrained(config.ASR_MODEL_PATH)

    def __call__(self, audio_data):
        # 序列化
        input_values = self.processor(audio_data, return_tensors="pt", padding="longest",
                                      sampling_rate=16000).input_values
        # 解码
        predicted_ids = torch.argmax(self.model(input_values).logits, dim=-1)
        transcription = list(map(str.lower, self.processor.batch_decode(predicted_ids)))[0]
        return transcription


class Translator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATOR_MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATOR_MODEL_PATH)
        self.translation = pipeline('translation_zh_to_en', model=self.model, tokenizer=self.tokenizer, )

    def __call__(self, text):
        return self.translation(text)[0]['translation_text']


class FLACConverter:  # pylint: disable=too-few-public-methods
    """
    Class for converting a region of an input audio or video file into a FLAC audio file
    """

    def __init__(self, source_path, include_before=0.25, include_after=0.25):
        self.source_path = source_path
        self.include_before = include_before
        self.include_after = include_after

    def __call__(self, region):
        try:
            start, end = region
            start = max(0, start - self.include_before)
            end += self.include_after
            temp = tempfile.NamedTemporaryFile(suffix='.flac', delete=False)
            command = ["ffmpeg", "-ss", str(start), "-t", str(end - start),
                       "-y", "-i", self.source_path,
                       "-loglevel", "error", temp.name]
            use_shell = True if os.name == "nt" else False
            subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
            read_data = temp.read()
            temp.close()
            os.unlink(temp.name)
            return read_data

        except KeyboardInterrupt:
            return None


class SubtitleGenerator:

    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def which(program):
        """
        Return the path for a given executable.
        """

        def is_exe(file_path):
            """
            Checks whether a file is executable.
            """
            return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

        fpath, _ = os.path.split(program)
        if fpath:
            if is_exe(program):
                return program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return exe_file
        return None

    def extract_audio(self, rate=16000):
        """
        Extract audio from an input file to a temporary WAV file.
        """
        temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        if not os.path.isfile(self.filename):
            print("The given file does not exist: {}".format(self.filename))
            raise Exception("Invalid filepath: {}".format(self.filename))
        if not self.which("ffmpeg"):
            print("ffmpeg: Executable not found on machine.")
            raise Exception("Dependency not found: ffmpeg")
        command = ["ffmpeg", "-y", "-i", self.filename,
                   "-ac", '1', "-ar", str(rate),
                   "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        subprocess.check_output(command, stdin=open(os.devnull), shell=use_shell)
        return temp.name, rate

    @staticmethod
    def percentile(arr, percent):
        """
        Calculate the given percentile of arr.
        """
        arr = sorted(arr)
        index = (len(arr) - 1) * percent
        floor = math.floor(index)
        ceil = math.ceil(index)
        if floor == ceil:
            return arr[int(index)]
        low_value = arr[int(floor)] * (ceil - index)
        high_value = arr[int(ceil)] * (index - floor)
        return low_value + high_value

    def find_speech_regions(self, filename, frame_width=4096, min_region_size=0.5,
                            max_region_size=6):  # pylint: disable=too-many-locals
        """
        Perform voice activity detection on a given audio file.
        """
        reader = wave.open(filename)
        sample_width = reader.getsampwidth()
        rate = reader.getframerate()
        n_channels = reader.getnchannels()
        chunk_duration = float(frame_width) / rate

        n_chunks = int(math.ceil(reader.getnframes() * 1.0 / frame_width))
        energies = []
        for _ in range(n_chunks):
            chunk = reader.readframes(frame_width)
            energies.append(audioop.rms(chunk, sample_width * n_channels))
        threshold = self.percentile(energies, 0.2)
        elapsed_time = 0

        regions = []
        region_start = None

        for energy in energies:
            is_silence = energy <= threshold
            max_exceeded = region_start and elapsed_time - region_start >= max_region_size

            if (max_exceeded or is_silence) and region_start:
                if elapsed_time - region_start >= min_region_size:
                    regions.append((region_start, elapsed_time))
                    region_start = None

            elif (not region_start) and (not is_silence):
                region_start = elapsed_time
            elapsed_time += chunk_duration
        return regions

    def run(self, output=None,
            concurrency=config.DEFAULT_CONCURRENCY,
            subtitle_file_format=config.DEFAULT_SUBTITLE_FORMAT):
        """
        Given an input audio/video file, generate subtitles in the specified language and format.
        """
        audio_filename, audio_rate = self.extract_audio()
        regions = self.find_speech_regions(audio_filename)
        pool = multiprocessing.Pool(concurrency)
        converter = FLACConverter(source_path=audio_filename)
        recognizer = AudioRecogniser()
        translator = Translator()

        transcripts = []
        if regions:
            try:
                extracted_regions = []
                translated_transcripts = []
                for i, extracted_region in enumerate(pool.imap(converter, regions)):
                    data, sr = librosa.load(io.BytesIO(extracted_region), sr=16000)
                    extracted_regions.append(data)

                for i, data in enumerate(extracted_regions):
                    transcript = recognizer(data)
                    print(transcript)
                    transcripts.append(transcript)
                    translated_transcript = translator(transcript)
                    print(translated_transcript)
                    translated_transcripts.append(translated_transcript)
                    print()

            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                print("Cancelling transcription")
                raise

        timed_subtitles = [(r, t) for r, t in zip(regions, transcripts) if t]
        formatter = FORMATTERS.get(subtitle_file_format)
        print(timed_subtitles)
        formatted_subtitles = formatter(subtitles=timed_subtitles)
        dest = output
        if not dest:
            base = os.path.splitext(self.filename)[0]
            dest = "{base}.{format}".format(base=base, format=subtitle_file_format)

        with open(dest, 'wb') as output_file:
            output_file.write(formatted_subtitles.encode("utf-8"))
        os.remove(audio_filename)
        return dest


if __name__ == '__main__':
    wav_path = os.path.join(os.path.dirname(__file__), 'test', 'test.flv')
    sg = SubtitleGenerator(wav_path)
    ret = sg.run()
    print(ret)
