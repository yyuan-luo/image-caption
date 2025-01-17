import spacy

en_tokenizer = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, frequency_threshold):
        # padding, start of sentence, end of sentence, unknown token
        self.stoi = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': 3}
        self.itos = {0: '[PAD]', 1: '[SOS]', 2: '[EOS]', 3: '[UNK]'}
        self.frequency_threshold = frequency_threshold

    def __len__(self):
        return len(self.stoi)

    @staticmethod
    def tokenize_sentence(sentence):
        return [token.text.lower() for token in en_tokenizer.tokenizer(sentence)]

    """
    Go through all sentences and collect any words appear 
    more than the given frequency into the vocabulary dictionary
    """
    def build_vocabulary(self, sentences):
        frequency_dict = {}
        token_idx = self.__len__()
        for sentence in sentences:
            for token in self.tokenize_sentence(sentence):
                if token not in frequency_dict:
                    frequency_dict[token] = 1
                else:
                    frequency_dict[token] += 1
                if frequency_dict[token] == self.frequency_threshold:  # "==" so that the token won't be duplicated
                    self.stoi[token] = token_idx
                    self.itos[token_idx] = token
                    token_idx += 1

    def numericalize(self, sentence):
        tokens = self.tokenize_sentence(sentence)
        return [self.stoi[token] if token in self.stoi else self.stoi['[UNK]'] for token in tokens]
