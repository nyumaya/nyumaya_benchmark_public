from pydub import AudioSegment
from config import samplerate

def load_audio_file(filename):

	sound = None

	if filename.endswith('.mp3') or filename.endswith('.MP3'):
		sound = AudioSegment.from_mp3(filename)
	elif filename.endswith('.wav') or filename.endswith('.WAV'):
		sound = AudioSegment.from_wav(filename)
	elif filename.endswith('.ogg'):
		sound = AudioSegment.from_ogg(filename)
	elif filename.endswith('.flac'):
		sound = AudioSegment.from_file(filename, "flac")

	sound = sound.set_frame_rate(samplerate)
	sound = sound.set_channels(1)
	sound = sound.set_sample_width(2)
	duration = sound.duration_seconds

	return sound,duration


extension_list=[".wav",".WAV",".mp3",".MP3",".flac",".ogg"]
