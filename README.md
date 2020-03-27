Intent Classification and Out-Of-Scope Prediction

Here we have the code to test the accuracies and recall for four intent classification models: Convolutional Neural Network, BERT, MultiLayer Perceptron and Dialogflow.

CNN: Convolutional Neural Network is known for its edge case detectability and with poper assignement of hyperparameters we can achive the best accuracy. The above file CNN.ipynb can be used for all the datasets and parameters are to be tuned for each dataset.

Dialogflow: project_agent is a basic dialogue flow agent. data should be imported to the agent and then it can be used for intent recognition.

BERT.ipynb- has Bert Algorithm Implemented for in-scope queries on small dataset. 

MLP.py - has Multi-Layer Perceptron Algorithm Implemented on small dataset for in-scope queries. Out-of-scope recall is also calculated for this algorithm

Data_Preprocess.py, Model_Classification.py and Find_Accuracies.py- are implemented to enable integration of all algorithms. 
In this phase, we only integrated MLP algorithms with these files. All other Algorithm files run independently. 
However, in the next phase we would integrate all algorithms. 
Data_Preprocess.py- has implementation required for preprocessing of data.To allow reusability of pre-processing code for all algorithms
Model_Classification.py- allows user to pik the classification algorithm of their choice
Find_Accuracies.py- to find accuracies for the algorithms on all datasets
