data_dir = '/SMT_DRUMS'
num_drums = 3
drums = ['HH', 'KD', 'SD', 'MIX']
num_gen_type = 3
gen_type = ['RealDrum', 'TechnoDrum_02', 'WaveDrum_02']

max_audio_length = 1323000
sample_rate = 44100.0

n_fft = 4096
win_length = 2048
hop_length = int(win_length/16.0)
window='hann'


test_filepath = 'test.wav'