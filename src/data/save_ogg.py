# Save audio numpy array to pickle file for debugging/analysis
import io
import logging
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
from platformdirs import user_data_dir
from scipy.io import wavfile

from src.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.setLevel(logging.DEBUG)

# Get FFmpeg path from ffdl installation
FFMPEG_PATH = Path(user_data_dir("ffmpeg-downloader")) / "ffmpeg" / "bin" / "ffmpeg.exe"
FFMPEG_PATH = Path(rf"C:\Users\andre\scoop\apps\ffmpeg\current\bin\ffmpeg.exe")
if not FFMPEG_PATH.exists():
    raise RuntimeError(
        f"FFmpeg not found at {FFMPEG_PATH}. Please run 'ffdl install' first."
    )


def save_ogg_via_wav(fs, audio, output_filepath):

    # Create a unique temporary filename for multiprocessing safety
    temp_wav_path = Path(f"./temp_{os.getpid()}_{time.time_ns()}.wav").absolute()

    try:
        start_time = time.perf_counter()

        # Ensure audio is normalized float32 between -1 and 1
        audio_normalized = audio.astype(np.float32)
        # max_val = np.max(np.abs(audio_normalized))
        # if max_val > 1.0:
        # audio_normalized = audio_normalized / max_val

        # Convert to int16 for WAV
        audio_int16 = (audio_normalized * 32767).astype(np.int16)

        # Save as temporary WAV file using scipy
        wavfile.write(str(temp_wav_path), fs, audio_int16)
        # logger.info(f"Saved temporary WAV file: {temp_wav_path}")

        # Convert to OGG using FFmpeg directly
        try:
            ffmpeg_cmd = [
                str(FFMPEG_PATH),
                "-y",  # Overwrite output file if it exists
                "-loglevel",
                "error",  # Suppress warnings and info messages
                "-i",
                str(temp_wav_path),
                "-c:a",
                "libvorbis",  # Use Vorbis codec
                "-ar",
                str(fs),  # Set sample rate
                # "-q:a",
                # "6",  # Quality setting (0-10, 10 being highest)
                str(output_filepath),
            ]
            # logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")

            result = subprocess.run(
                ffmpeg_cmd, capture_output=True, text=True, check=True
            )
            logger.debug(f"FFmpeg stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"FFmpeg stderr: {result.stderr}")

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise

        # Clean up temporary file
        temp_wav_path.unlink()

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        # file_size = output_filepath.stat().st_size
        # logger.info(f"Audio saved to: {output_filepath}")
        # logger.info(f"File size: {file_size / 1024:.2f} KB")
        # logger.info(f"Processing time: {processing_time:.2f} seconds")
        # logger.info(
        # f"Processing speed: {(file_size / 1024 / 1024) / processing_time:.2f} MB/s"
        # )

    except Exception as e:
        logger.error(f"Error saving audio: {e}")
        if temp_wav_path.exists():
            temp_wav_path.unlink()


if __name__ == "__main__":
    fs = 32000
    output_path = Path("./sound.ogg").absolute()
    pickle_path = Path("./sound.pickle").absolute()
    with open(pickle_path, "rb") as f:
        audio = pickle.load(f)
    logger.info(f"Audio shape: {audio.shape}")
    save_ogg_via_wav(fs, audio, output_path)
