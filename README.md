# QuoteGenerator-Keras
A simple character level quote generator. It uses chunks of characters from a quote and uses the next character of the sequence as output token. Varying the lookback amount changes the structure of generated quotes considerably. Model consists of stack of LSTM layers and outputs a one-hot representation of tokens.


# Dependencies
* [Keras (Tensorflow Backend)](https://keras.io/)
* Numpy
* Pandas
* [Jupyter](http://jupyter.readthedocs.io/en/latest/install.html)

# Usage
For local usage/training:
1. Download and install Jupyter Notebook and IPython kernel
1. run a Jupyter environment locally using ``` jupyter notebook ``` in the terminal
1. call ``` load_model() ``` to load pretrained model in /models dir and train model 
1. change temperature of output(random sampling coefficient) in ```sample()``` <br>
   (higher value of temperature = higher randomness, lower value of temperature preserves local structure but increases redundancy)
1. run ``` makeinference.py ``` with starting phrase and quote length 

For working on Google colab:<br>
* ```!git clone https://github.com/ArthDh/QuoteGenerator-Keras```

# Example Output
```
Enter a starting phrase: 	
Enter length of quote to be produced: 50
------ temperature: 0.1
 i can see the best of the states of the courage to
------ temperature: 0.5
 i car. i know i've have a lot of change. 
ell that
------ temperature: 0.7
 indiving a long because i make a society is to lea
------ temperature: 1.0
 inget providing hard again. 
ends in the hot of th
------ temperature: 1.2
 i'moke is goer that really everybody's ltear' live
 
 Enter a starting phrase: 		
Enter length of quote to be produced: 100
------ temperature: 0.1
 i want to be a strong to the courage to be a simple and the production of the same things that i was
------ temperature: 0.5
 i got a day and be a seemer of the life for the best way, when i was a new pretty to much other than
------ temperature: 0.7
 a dono  you're a very song what i am individual even satisfy, and the family proud in a production. 
------ temperature: 1.0
 incase his to way and case. 
ell i would be f but be a day to exergited me. but i had eithere of thi
------ temperature: 1.2
 igreat two year. 
evelopming to celebr space humall for vuices on heigh things.' you're always keepl

```

# References
* [Dataset](https://www.researchgate.net/publication/304742521_CSV_dataset_of_76000_quotes_suitable_for_quotes_recommender_systems_or_other_analysis)
* Inspired by [Jabrils' Quote generator](https://twitter.com/SEFDStuff) 

<h6> * Probably could use some more training. </h6>
