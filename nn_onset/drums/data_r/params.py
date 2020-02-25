sample_rate = 44100
spec_hop_length = 512
spec_fmin = 0
spec_n_bins = 229
n_fft = 2048
batch_size = 8

bols =['Da', 'Ke', 'Ge', 'Ta ','Ti', 'Na', 'Din', 'Kda', 'Tit', 'Dha', 'Dhere', 'Dhit', 'Dhi', 'Dhin', 'Re', 'Te', 'Tun', 'Tin', 'Traka']
num_bols = len(bols)

rain_data_dir = '../data/np_dir/'
train_spec_dir = train_data_dir + 'spec/'
train_onset_dir = train_data_dir + 'onset/'
train_bols_dir = train_data_dir + 'bols/'


train_wav_dir = '../train_data/wav/'
train_anott_dir = '../train_data/trans/'

# [('activation_loss', False)
# , ('batch_size', 8)
# , ('clip_norm', 3)
# , ('combined_lstm_units', 128)
# , ('cqt_bins_per_octave', 36)
# , ('decay_rate', 0.98)
# , ('decay_steps', 10000)
# , ('frame_bidirectional', True)
# , ('frame_lstm_units', 0)
# , ('jitter_amount_ms', 0)
# , ('jitter_wav_and_label_separately', False)
# , ('learning_rate', 0.0006)
# , ('min_duration_ms', 0)
# , ('min_frame_occupancy_for_label', 0.0)
# , ('normalize_audio', False)
# , ('onset_bidirectional', True)
# , ('onset_delay', 0)
# , ('onset_length', 32)
# , ('onset_lstm_units', 128)
# , ('onset_mode', 'length_ms')
# , ('sample_rate', 16000)
# , ('share_conv_features', False)
# , ('spec_fmin', 30.0)
# , ('spec_hop_length', 512)
# , ('spec_log_amplitude', True)
# , ('spec_n_bins', 229)
# , ('spec_type', 'mel')
# , ('stop_activation_gradient', False)
# , ('stop_onset_gradient', False)
# , ('truncated_length', 1500)
# , ('velocity_lstm_units', 0)
# , ('weight_frame_and_activation_loss', True)
# ])
