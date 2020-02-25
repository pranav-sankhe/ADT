# Final Presentation Notes 

## Slide 3


### Structure of Bi-LSTMs
LSTM in its core, preserves information from inputs that has already passed through it using the hidden state.
Unidirectional LSTM only preserves information of the past because the only inputs it has seen are from the past. Using bidirectional will run your inputs in two ways, one from past to future and one from future to past and what differs this approach from unidirectional is that in the LSTM that runs backwards you preserve information from the future and using the two hidden states combined you are able in any point in time to preserve information from both past and future.

### Pre-training RNN
With transfer learning, we can take a pretrained model, which was trained on a large readily available dataset (trained on a completely different task, with the same input but different output). We use the output of that layer as input features to train a much smaller network that requires a smaller number of parameters. This smaller network only needs to learn the relations for your specific problem having already learnt about patterns in the data from the pretrained model. This way a model trained to detect Cats can be reused to Reproduce the work of Van Gogh



### Problems with pre-training RNN
Pre-Training requires the input to be same. The pretraining dataset was skeletal data with 25 joints. Our data has 104 joints. The joints which are common between both the datasets were 12 which were very low. 
We trained the RNN network shown in figure on the pretraining datasets considering only the intersecting joints but the results were not impressive. We could achieve a bare accuracy of 15%. This explained why pre-training on RNNs is not very common as against convolutional neural networks. 
Hence we had to drop this model. 


## Slide 4

### Variance based feature selection 
In order to understand and preprocess the data, we performed varaince based analysis to handpick the features. We observed that some of the joints had almost zero movement and hence including them would not make sense.
### C3D data visualization 
During the process of selecting which data should we use, we explored the C3D format. This is the visualization of point cloud data stored in C3D format.     
### Intersecting joints between our dataset and the pre-training dataset


## Slide 5

### Motivation to use Optical Flow
Optical Flow is one of the most widely used methods to extract motion information. Optical Flow is a very high level motion feature which is tough
for a neural network to learn on having videos as inputs. The results in activity recognition tasks have been highly improvised after including optical flow as inputs in the architecture.    

### Optical Flow is the apparent motion of brightness patterns in the image
Optical Flow assumes that a pixel's intensity doesn't change over small interval of time[betweeen two frames]. It creates a vector field of displacement vectors for motion of each pixel.  
### Calculated Optical Flow and separated the X and Y components 
We can seperate the Optical Flow vector field in two orthogonal components.   
### Two channeled Optical Flow data input
The optical flow input to the model was 2 channeled. The first channel being the optical flow in x direction and the second one being the optical flow in y direction. The input dimensions were: [BATCH_SIZE, NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS]



## Slide 6

The model now looked like this. We used a sequence to sequence model where the encoder was inspired from state of art action recognition models. The CNN is supposed to learn the spatial features of the motion and the RNNs are supposed to learn the temporal features. To preserve the temporal information the output from the last CNN layer was divided in segments and then was fed to RNN which gives a fixed vector as an output which is called as context vector in the sequence to sequence terminology. 

The decoder model is a simple 2 layer RNN model as used in the standard NMT models. 


## Slide 7

We added residual blocks as described in the resnet paper after taking inspiration from the current state of art activity classification model which uses ResNet. After adding residual blocks we were able to make the model more deep and easier to train.

The reason why we didn't use the RGB videos and just the optical flow was that the data we had was just of single person and also we wanted to keeo the model simple and adding a two stream input would have made it more complex and harder to train.

Instead of learning a direct mapping of x ->y with a function H(x) (A few stacked non-linear layers). Let us define the residual function using F(x) = H(x) — x, which can be reframed into H(x) = F(x)+x, where F(x) and x represents the stacked non-linear layers and the identity function(input=output) respectively.

If the identity mapping is optimal, We can easily push the residuals to zero (F(x) = 0) than to fit an identity mapping (x, input=output) by a stack of non-linear layers. In simple language it is very easy to come up with a solution like F(x) =0 rather than F(x)=x using stack of non-linear cnn layers as function (Think about it). So, this function F(x) is what the authors called Residual function. 

How resnet solves the problem of vanishing gradients

More layers is better but because of the vanishing gradient problem, model weights of the first layers can not be updated correctly through the backpropagation of the error gradient (the chain rule multiplies error gradient values lower than one and then, when the gradient error comes to the first layers, its value goes to zero).
That is the objective of Resnet : preserve the gradient.
How ? Thanks to the idendity matrix because “what if we were to backpropagate through the identity function? Then the gradient would simply be multiplied by 1 and nothing would happen to it!”.

## Slide 8 

The training was planned in two stages.
Since we had less data [around 1300 video files], we planned to pre-train the encoder model on the huge public datasets and use some initial pre-trained layers directly in the encoder. The remaining model would be trained on our collected dataset so as to tune the model for our specific task. 

The problem with implementing this approach was that calculating Optical Flow is a heavy computation and takes of time. It would have taken a considerable time to convert the pre-training datasets to optical flow image frames and hence we planned to focus on converting the collected data to optical flow and make the model train on it as of now and hold on the pre-training for a while. 



## Slide 9

While you're training your encoder-decoder models (and once you have trained models), you can obtain translations given previously unseen source sentences. This process is called inference. There is a clear distinction between training and inference (testing): at inference time, we only have access to the source sentence, i.e., encoder_inputs. There are many ways to perform decoding. Decoding methods include greedy, sampling, and beam-search decoding. 

Greedy Search selects the most likely word at each step in the output sequence.

Another popular heuristic is the beam search that expands upon the greedy search and returns a list of most likely output sequences.

Instead of greedily choosing the most likely next step as the sequence is constructed, the beam search expands all possible next steps and keeps the k most likely, where k is a user-specified parameter and controls the number of beams or parallel searches through the sequence of probabilities.

The local beam search algorithm keeps track of k states rather than just one. It begins with k randomly generated states. At each step, all the successors of all k states are generated. If any one is a goal, the algorithm halts. Otherwise, it selects the k best successors from the complete list and repeats.



## Appendix

### Encoder Decoder Model 

An Encoder-Decoder architecture was developed where an input sequence was read in entirety and encoded to a fixed-length internal representation.
A decoder network then used this internal representation to output words until the end of sequence token was reached. LSTM networks were used for both the encoder and decoder.

The encoder maps a variable-length source sequence to a fixed-length vector, and the decoder maps the vector representation back to a variable-length target sequence.



 