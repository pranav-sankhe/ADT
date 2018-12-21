gen_type = ['RealDrum', 'TechnoDrum', 'WaveDrum']
data_dir = '../data/SMT_DRUMS'
num_drums = 3

n_fft = 4096	#92 milliseconds 
win_length = 2048	#46 milliseconds 
hop_length = int(win_length/16.0)	#3 ms	
window='hann'
sample_rate = 44100


get_template_length = 1
test_filepath = '/Users/sabsathai/Documents/projects/drum_transcription/data/test.wav'

HH_Ncomponents = 2
KD_Ncomponents = 2
SD_Ncomponents = 2

solver = 'cd'#'mu'
max_iter = 200
alpha = 0.5