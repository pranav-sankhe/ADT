onset_bidirectional = True 
activation_loss = False
onset_lstm_units = 128
velocity_lstm_units = 128
share_conv_features = False  # this is broken when hparams.frame_lstm_units > 0
frame_lstm_units = 128
stop_activation_gradient=False
stop_onset_gradient=False
combined_lstm_units = 128
frame_bidirectional = True
weight_frame_and_activation_loss = False
learning_rate=0.0006
max_gradient_norm = 10

sample_rate = 44100
freq_length = 229
batch_size = 8
time_len = 1723
truncated_length = 1723
decay_steps=10000
decay_rate=0.98
hop_length = 512
num_iterations = 1000

batch_size = 8

bols =['Da', 'Ke', 'Ge', 'Ta ','Ti', 'Na', 'Din', 'Kda', 'Tit', 'Dha', 'Dhere', 'Dhit', 'Dhi', 'Dhin', 'Re', 'Te', 'Tun', 'Tin', 'Traka']
num_bols = len(bols)



log_dir = './train'

train_data_dir = '../data/np_dir/'
train_spec_dir = train_data_dir + 'spec/'
train_onset_dir = train_data_dir + 'onset/'
train_bols_dir = train_data_dir + 'bols/'
