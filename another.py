import os
import numpy as np
from pydub import AudioSegment
import soundfile as sf
from scipy.signal import lfilter
import wave
from tempfile import NamedTemporaryFile
from typing import Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AudioConfig:
    """Configuration class for 8D audio conversion parameters."""
    pan_speed: float = 0.5  # Speed of panning (Hz)
    depth: float = 0.95     # Depth of the panning effect (0-1)
    reverb_delay: int = 50  # Reverb delay in milliseconds
    reverb_decay: float = 0.3  # Reverb decay factor (0-1)

class Audio8DConverter:
    """Class to convert regular audio to 8D audio effect."""
    
    def __init__(self, config: AudioConfig = None):
        """Initialize the 8D audio converter with optional configuration."""
        self.config = config if config is not None else AudioConfig()
        self._validate_config()
        self._check_ffmpeg()

    def _check_ffmpeg(self) -> None:
        """Check if ffmpeg is installed and accessible."""
        try:
            # Try multiple possible ffmpeg paths
            ffmpeg_paths = [
                "ffmpeg",
                "/usr/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
                "/opt/homebrew/bin/ffmpeg"
            ]
            
            for path in ffmpeg_paths:
                result = os.system(f"{path} -version > /dev/null 2>&1")
                if result == 0:
                    logger.info("FFmpeg is installed and accessible.")
                    return
    
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "Windows: choco install ffmpeg\n"
                "Mac: brew install ffmpeg\n"
                "Linux: sudo apt-get install ffmpeg"
            )
        except Exception as e:
            logger.error(f"FFmpeg check failed: {str(e)}")
            raise

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.config.pan_speed <= 2:
            raise ValueError("Pan speed must be between 0 and 2 Hz")
        if not 0 <= self.config.depth <= 1:
            raise ValueError("Depth must be between 0 and 1")
        if not 0 < self.config.reverb_delay <= 100:
            raise ValueError("Reverb delay must be between 0 and 100 ms")
        if not 0 <= self.config.reverb_decay <= 1:
            raise ValueError("Reverb decay must be between 0 and 1")

    def _apply_panning(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply panning effect to the audio."""
        duration = len(audio_data) / sample_rate
        t = np.linspace(0, duration, len(audio_data))
        pan_curve = np.sin(2 * np.pi * self.config.pan_speed * t) * self.config.depth

        if len(audio_data.shape) == 1:
            audio_data = np.stack([audio_data, audio_data], axis=1)

        left_channel = audio_data[:, 0] * (1 + pan_curve)
        right_channel = audio_data[:, 1] * (1 - pan_curve)
        
        return np.stack([left_channel, right_channel], axis=1)

    def _apply_reverb(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply reverb effect to the audio."""
        delay_samples = int(self.config.reverb_delay * sample_rate / 1000)
        decay = self.config.reverb_decay

        ir = np.zeros(delay_samples)
        ir[0] = 1
        ir[delay_samples-1] = decay

        audio_reverb = np.zeros_like(audio_data)
        for i in range(audio_data.shape[1]):
            audio_reverb[:, i] = lfilter(ir, [1], audio_data[:, i])

        return audio_data + audio_reverb * 0.5

    def convert_file(self, input_path: str, output_path: str) -> bool:
        """Convert a regular MP3 file to 8D audio."""
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            if not input_path.lower().endswith('.mp3'):
                raise ValueError("Input file must be an MP3 file")

            logger.info(f"Converting file: {input_path}")

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Load audio file using pydub
            audio = AudioSegment.from_mp3(input_path)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            # Handle both mono and stereo
            if audio.channels == 1:
                samples = samples.reshape(-1, 1)
            else:
                samples = samples.reshape(-1, 2)
            
            # Convert to float32 and normalize
            audio_data = samples.astype(np.float32) / np.iinfo(samples.dtype).max

            # Apply effects
            audio_data = self._apply_panning(audio_data, audio.frame_rate)
            audio_data = self._apply_reverb(audio_data, audio.frame_rate)

            # Normalize to prevent clipping
            audio_data = audio_data / np.max(np.abs(audio_data))

            # Scale back to int16
            audio_data = (audio_data * 32767).astype(np.int16)

            # Save as temporary WAV first
            with NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wav_file:
                    wav_file.setnchannels(2)  # Always stereo output
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(audio.frame_rate)
                    wav_file.writeframes(audio_data.tobytes())

                # Convert WAV to MP3 using pydub
                AudioSegment.from_wav(temp_wav.name).export(
                    output_path, 
                    format='mp3',
                    bitrate='320k'
                )

            # Clean up temporary file
            os.unlink(temp_wav.name)
            
            logger.info(f"Successfully converted file: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error converting file: {str(e)}")
            raise

def convert_to_8d(input_file: str, output_file: str, config: Optional[AudioConfig] = None) -> bool:
    """Convenience function to convert an MP3 file to 8D audio."""
    converter = Audio8DConverter(config)
    return converter.convert_file(input_file, output_file)
