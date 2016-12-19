#Program overview
This github contains two different python script: one to train the neural network model and one to test it against a given data sets.
The neural network is a recurrent neural network (LTSM) of 3 layer each with 256 nodes.
The network takes in a 256^2 sized array which represents an equivalent spectrogram of the spoken word audio file. The data is already preprocessed.
The neural network outputs the first letter of the letter. The resulted accuracy of the best neural network configuration is 40%.


Program dependencies: Tensorflow and scikit-image

To run the predictions on a trained model: run test-deep-net.py
	Outputs will be averages for N number of batches
	
To run training algorithm:
	You will need to acquire the data set at https://www.dropbox.com/s/2ff0x8z60bjjz7b/spoken_words.tar?dl=0
	Extract all the data into a folder call spoken_words relative to your current directory of deep-net.py
	run deep-net.py