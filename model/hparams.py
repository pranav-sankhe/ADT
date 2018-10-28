gen_type = ['RealDrum', 'TechnoDrum', 'WaveDrum']
data_dir = '../data/SMT_DRUMS'
num_drums = 3

n_fft = 4096
win_length = 2048
hop_length = 3*win_length/4
window='hann'
sample_rate = 44100

max_audio_length = 1323000

get_template_length = 1
test_filepath = '/Users/sabsathai/Documents/projects/drum_transcription/data/test.wav'