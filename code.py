!pip install pandas numpy scikit-learn

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('HMM.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_sentences = train_data['Sentence']
train_tags = train_data['POS_Tags']

test_sentences = test_data['Sentence']
test_tags = test_data['POS_Tags']

train_words = [sentence.split() for sentence in train_sentences]
train_pos_tags = [tags.split() for tags in train_tags]

test_words = [sentence.split() for sentence in test_sentences]
test_pos_tags = [tags.split() for tags in test_tags]

class HMM_POS_Tagger:
    def __init__(self):
        self.transition_prob = defaultdict(lambda: defaultdict(int))
        self.emission_prob = defaultdict(lambda: defaultdict(int))
        self.pos_tags = set()
        self.words = set()

    def train(self, sentences, pos_tags):
        for sentence, tags in zip(sentences, pos_tags):
            sentence = ['<START>'] + sentence + ['<END>']
            tags = ['<START>'] + tags + ['<END>']

            for i in range(1, len(sentence)):
                word, tag = sentence[i], tags[i]
                self.words.add(word)
                self.pos_tags.add(tag)
                self.emission_prob[tag][word] += 1
                self.transition_prob[tags[i-1]][tag] += 1

    def normalize_probs(self):
        for tag, words in self.emission_prob.items():
            total_count = sum(words.values())
            for word in words:
                self.emission_prob[tag][word] /= total_count

        for prev_tag, tags in self.transition_prob.items():
            total_count = sum(tags.values())
            for tag in tags:
                self.transition_prob[prev_tag][tag] /= total_count

    def predict(self, sentence):
        sentence = ['<START>'] + sentence + ['<END>']
        n = len(sentence)

        dp = np.zeros((n, len(self.pos_tags)))
        backpointer = np.zeros((n, len(self.pos_tags)), dtype=int)

        for i, tag in enumerate(self.pos_tags):
            dp[0, i] = np.log(self.emission_prob[tag].get(sentence[0], 1e-6)) + np.log(self.transition_prob['<START>'].get(tag, 1e-6))

        for t in range(1, n):
            for i, tag in enumerate(self.pos_tags):
                max_prob = -float('inf')
                max_state = -1
                for j, prev_tag in enumerate(self.pos_tags):
                    prob = dp[t-1, j] + np.log(self.transition_prob[prev_tag].get(tag, 1e-6)) + np.log(self.emission_prob[tag].get(sentence[t], 1e-6))
                    if prob > max_prob:
                        max_prob = prob
                        max_state = j
                dp[t, i] = max_prob
                backpointer[t, i] = max_state


        tags = []
        best_tag_idx = np.argmax(dp[n-1])
        for t in range(n-1, -1, -1):
            tags.append(self.pos_tags[best_tag_idx])
            best_tag_idx = backpointer[t, best_tag_idx]
        tags.reverse()
        return tags[1:-1]

hmm_tagger = HMM_POS_Tagger()

hmm_tagger.train(train_words, train_pos_tags)

hmm_tagger.normalize_probs()

predicted_tags = [hmm_tagger.predict(sentence) for sentence in test_words]

flattened_test_tags = [tag for tags in test_pos_tags for tag in tags]
flattened_predicted_tags = [tag for tags in predicted_tags for tag in tags]

accuracy = accuracy_score(flattened_test_tags, flattened_predicted_tags)
print(f"Accuracy: {accuracy * 100:.2f}%")
