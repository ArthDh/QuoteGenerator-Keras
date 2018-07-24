import keras
import sys
import pickle
import numpy as np
from keras.models import load_model


def match_input(sent, input_len=20):
    '''
    Inference is a helper function that
    resizes text to match input layer in model

    returns string of required size
    '''
    string_length = input_len  # Input length - 20 for QuoteGen
    string_revised = sent.ljust(string_length)
    return string_revised


def sample(preds, temperature=0.2):
    '''
    Used to introduce entropy in sampling
    Higher the temperature more the samping randomness

    returns maximum of reweighted predictions
    '''
    preds = np.reshape(preds, preds.shape[-1])
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-25) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def make_inference(start_word, model, max_len, word_index, index2char, s_len):
    """
    start_word: starting phrase
    model: model used for prediction
    max_len: Input length for model
    word_index: dictionary mapping for tokens
    index2char: reverse dictionary mapping for tokens
    s_len: sampled sentence length

    """
    start_word = start_word
    inference_text = match_input(start_word, max_len)

    for temperature in [0.1, 0.5, 0.7, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(inference_text.strip() + " ")
        generated_text = inference_text[:20] + ""
        for i in range(s_len):
            sampled = np.zeros((1, max_len, len(word_index)))
            for t, char in enumerate(generated_text):
                sampled[0, t, word_index[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = index2char[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
        print()


def main():
    max_len = 20
    word_index = {}
    index2char = {}

    model_path = "" # Path to model
    model = load_model(model_path)

    # Open word_index.pkl
    with open('', 'rb') as handle:
        word_index = pickle.load(handle)

    for ch in word_index:
        index2char[word_index.get(ch)] = ch

    f = 1
    while f == 1:
        start_word = str(input("Enter a starting phrase: "))
        s_len = int(input("Enter length of quote to be produced: "))
        make_inference(start_word, model, max_len, word_index, index2char, s_len)
        f = int(input("Press 1 to continue.."))


if __name__ == '__main__':
    main()
