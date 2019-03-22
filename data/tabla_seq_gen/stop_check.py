mixed_res=['Dha', 'Dhin']
mixed_damp=['Dhere', 'Dhi', 'Dhit']
treble_res=['Na','Tin','Tun','Din']
treble_damp=['Ti','Ta','Te','Re','Tak','Tit','Da','Traka']
bass_res=['Ge']
bass_damp=['Ke']

prev_bol=''; curr_bol=''

#Conditions when previous stroke needs to be stopped:
if (((prev_bol in mixed_res) & (curr_bol in (mixed_res+mixed_damp))) | ((prev_bol in mixed_damp) & (curr_bol in (bass_res+bass_damp+mixed_damp+mixed_res))) | ((prev_bol in treble_res) & (curr_bol in (treble_res+treble_damp+mixed_res+mixed_damp))) | ((prev_bol in bass_res) & (curr_bol in (bass_res+mixed_res+mixed_damp)))):
	print('Alert')
