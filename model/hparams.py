gen_type = ['RealDrum', 'TechnoDrum', 'WaveDrum']
data_dir = '/Users/sabsathai/Documents/projects/drum_transcription/data/SMT_DRUMS'
num_drums = 3

n_fft = 4096
hop_length = 64
win_length = 256
window='hann'
sample_rate = 44100

max_audio_length = 1323000

get_template_length = 1
test_filepath = '/Users/sabsathai/Documents/projects/drum_transcription/data/test.wav'