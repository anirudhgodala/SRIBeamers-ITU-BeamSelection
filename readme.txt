Description: The files in this folder are submission files by SRIBeamers Team that uses the baseline data with a customized frontend.

1 - Python Dependencies
	- Python packages: tensorflow-gpu 2.3.1, Numpy

2 - Running the code
	2.1 - Training the model

	python beam_train_model.py --path ~/Documents/Raymobtime_dataset/s008/baseline_data/

	usage: beam_train_model.py 
							   [--path ]
							   data_folder

	Configure the files before training the net.
	positional arguments:
	  data_folder           Location of the data directory
	
	2.2 - Testing the model
	
	python beam_test_model.py --path ~/Documents/Raymobtime_dataset/s010/baseline_data/

	usage: beam_train_model.py 
							   [--path ]
							   data_folder

	Configure the files before training the net.
	positional arguments:
	  data_folder           Location of the data directory

3 - Pre-trained model and weights

	As described in the rules of the challenge, we also include the files my_model_50_drop.json and my_model_weights_50_drop.h5.
	These files are the results of running the training described in 2.1 with the
	following arguments:
	"python beam_train_model.py --path ~/Documents/Raymobtime_dataset/s008/baseline_data/".
