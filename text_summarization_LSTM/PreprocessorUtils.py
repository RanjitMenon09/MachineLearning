from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
class PreprocessorUtils:
    """
    class for preprocessing utils
    """
    def __init__(self):
        self.tokenizer = None
        self.max_length_combined = 824 #this is the max length of dialog column in dataset

    # #1 - Tokenize the training data
    def fit_tokenizer(self, train_dialogues, train_summaries, tokenizer_path):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(train_dialogues + train_summaries)

        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer saved successfully!")
        return self.tokenizer

    # #2 - Convert training data to numerical form
    def convert_to_sequences(self, df):
        df['dialogue'] = self.tokenizer.texts_to_sequences(df['dialogue'])
        df['summary'] = self.tokenizer.texts_to_sequences(df['summary'])
        return df

    # #3 - Print corresponding words from sequences
    def print_sequence_words(self, sequence):
        for num in sequence:
            word = self.tokenizer.index_word.get(num, 'UNK')
            print(word, end=' ')
        print()  # for newline

    # #4 - Pad the sequences
    def pad_sequences(self, dialogues, summaries):
        padded_dialogues = pad_sequences(dialogues, maxlen=self.max_length_combined, padding='post')
        padded_summaries = pad_sequences(summaries, maxlen=self.max_length_combined, padding='post')
        return padded_dialogues, padded_summaries
