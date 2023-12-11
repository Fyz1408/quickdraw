# quickdraw
Small little project to use Googles QuickDraw data set

## Setup
1) Get the dataset from Google Quick Draw and place the .npy files in /data folder.
2) Then run LoadData.py which will load all the data from the /data folder and store the features and labels in pickel files.
3) Now we need to train the model so run `QD_trainer.py` which will load data from pickle and augment it and after this, the training process begins.
4) At last run `main.py` to run the program and start drawing

## Thanks to Akshay Bahadur
I've drawn some inspiration from his code check him out [here](https://github.com/akshaybahadur21)
