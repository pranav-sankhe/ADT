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
weight_frame_and_activation_loss = True


bols = ['DA', 'KI', 'GE', 'TA', 'NA', 'DIN', 'KDA', 'TIT', 'DHA', 'DHE', 'DHET', 'DHI', 'DHIN', 'RE', 'TE', 'TII', 'TIN', 'TRA']
num_bols = len(bols)