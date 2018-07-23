import keras
import sys
import numpy as np
from keras.models import load_model

# max_len, word_index, index2char


def match_input(sent, input_len=20):
    '''
    Inference is a helper function that
    resizes text to match input layer in model

    returns string of required size
    '''
    string_length = input_len  # Input length - 20 for QuoteGen
    string_revised = sent.ljust(string_length)
    return string_revised


def make_inference(start_word, model):
    start_word = start_word
    inference_text = match_input(start_word)

    for temperature in [0.1, 0.5, 0.7, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(inference_text.strip() + " ")
        generated_text = inference_text[:20] + ""
        for i in range(50):
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
    model_path = "/Users/arth/Downloads/quotegen.h5"
    model = load_model(model_path)
    start_word = str(input("Enter a starting phrase: "))
    make_inference(start_word, model)


if __name__ == '__main__':
    main()
