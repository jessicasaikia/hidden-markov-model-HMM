# Hidden Markov Model (HMM)
This repository implements a Hidden Markov Model (HMM) for performing Parts-of-Speech (POS) Tagging on Assamese-English code-mixed texts. 

## Introduction to Parts-of-Speech (PoS) Tagging 
PoS tagging is the process of identifying and labeling grammatical roles of words in texts, supporting applications like machine translation and sentiment analysis. While different languages may have their own PoS tags, I have used my own custom PoS tags for this model. The Table below defines the custom PoS tags used in this model-

![Table](https://github.com/jessicasaikia/hidden-markov-model-HMM/blob/main/Custom%20PoS%20tags%20Table.png)

## About Hidden Markov Model (HMM)
The HMM is a statistical model that assumes that a system transitions between a series of hidden states based on probabilities. It is efficient in case of sequential data and POS tagging. 

The key components involved in its working are: 
- **States**: The hidden states represent the POS tags. These are underlying variables that can generate the observed data but are directly observable.
- **Observations**: These are the observed tokens in a sentence or the variables that can be measured and observed.
- **Transition probabilities**: It describes the probability of moving from one POS tag to another or from one hidden state to another.
- **Emission probabilities**: It gives the probability of a word being associated with a specific POS tag; thus, it describes the probability of observing an output given in a hidden state.

**Algorithm**:
1. The model imports the libraries and reads the dataset.
2. The transition matrix is initialised to store the transition probabilities and the emission matrix is initialised to store the emission probabilities.
3. Count frequencies for Transition and Emission matrices and update the transition and emission matrices.
4. Convert counts to probabilities
   - For each tag in transition matrix, normalise the counts by dividing each count by the total count of transitions from that tag, resulting in a probability distribution.
   - For each tag in emission matrix, normalise the counts by dividing each count by the total count of emissions for that tag.
5. Use a Viterbi algorithm to predict the POS tags by setting up the starting probabilities, then working through the sentence and lastly, back-tracing the best path to get the best sequence of tags.
6. For words not in emission matrix (words not seen in training), assign a small fixed probability. This helps avoid errors for unknown words.

## Where should you run this code?
I used Google Colab for this Model.
1. Simply create a new notebook (or file) on Google Colab.
2. Paste the code.
3. Upload your CSV dataset file to Google Colab.
4. Please make sure that you update the "path for the CSV" part of the code based on your CSV file name and file path.
5. Run the code.
6. The output will be displayed and saved as a different CSV file.

You can also VScode or any other platform (this code is just a python a code)
1. In this case, you will have to make sure you have the necessary libraries installed and dictionaries loaded correctly.
2. Simply run the program for the output.

## Additional Notes from me
In case of any help or queries, you can reach out to me in the comments or via my socials. My socials are:
- Discord: jessicasaikia
- Instagram: jessicasaikiaa
- LinkedIn: jessicasaikia (www.linkedin.com/in/jessicasaikia-787a771b2)

Additionally, you can find the custom dictionaries that I have used in this project and the dataset in their respective repositories on my profile. Have fun coding and good luck! :D
