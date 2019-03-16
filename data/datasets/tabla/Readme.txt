##########
Some notes
##########

Bol_mapping
###########
- The text file 'bol_mapping.txt' gives the mapping between all the aliases of every bol and the name it is referred to with in our isolated stroke samples - the first entry in each row is the name used in our isolated stroke samples, and the entries that follow are all the aliases found in the scores.

- Two special cases: Dhere and Traka
	The scores always have Dhe & Re or Tra & Ka marked as two separate bols, but our isolated strokes dataset has one sound sample for each of these combinations: Dhe+Re (called Dhere) and Tra+Ka(called Traka). This is because a Dhe is always followed by a Re and a Tra is followed by a Ka. So when you parse the score and read a 'Dhe' bol, you can read the next bol too(which will be a 'Re') and plant one sample of Dhere from our isolated strokes.
	Similarly for Traka.
 
Compositions
#############
- You might notice that the score text files have weird filenames, you can ignore that. They aren't of much significance.

- A semicolon marks the beat boundary - therefore all the bols between a pair of semicolons occur in one beat interval
	(So you should notice that all the scores have no less than 16 semicolons as each composition spans at least one cycle of 16 beats)

- A comma is used when two bols occur at double the speed - ie., two bols take the same duration as one bol
	e.g.: Taa Taa Ti,Ra Ki,Ta 
	Here, the bol 'Taa' occupies the same duration as the pair of bols 'Ti,Ra' or 'Ki,Ta'

- A hyphen indicates a pause of one bol duration (i.e., it is as long in time as a bol in its place would have been)

- To decide the BPM and generate the sound sequence, you can use one of the following two ways:
	1  I have also attached the transcriptions of the original audios from the UPF dataset (they contain onset time and bol label), so you could just use this file and generate a sound sequence by reading the onset time and bol label from the csv file, then mapping the bol label to one of our isolated strokes using the bol mapping, and then planting the sound at the onset time
	2 You could freely set a bpm between 60-100 and then use the score to find how many bols occur on every beat(between two semicolon marks) and then generate a sound sequence
	(You will have to use method 2 eventually to generate new sequences. Method 1 is just useful for the UPF dataset)
