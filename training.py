import pandas as pd
import torch
from torch import nn

import numpy as np
import csv
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from typing import Iterator, Iterable, List, Tuple, Text, Union, Sequence, Text

np.random.seed(42)

SEED = 42

##Code heavily borrowed from LING 582 class examples -  THANK YOU, Gus!!

###DATA PROCESSING#######################################################
def read_csvtagged(csvtagged_path: str):
    """
    Reads rows from a csv .tagged file.
    Each row consists of 3 columns of information:

    COLUMN	DESCRIPTION
    ID	Unique ID for this datapoint
    TEXT	Two snippets of text separated by [SNIPPET]
    LABEL	The label for this datapoint (see below)

    The labels are:
    0	Not the same author
    1	Same author
    """
    rows = []
    with open(csvtagged_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append((row[0], row[1], row[2]))
    return rows

everything = read_csvtagged('data/best_training_data.csv')[1:]
print(f"length of training data: {len(everything)}")


everything_x = [text for _, text, _ in everything]
everything_y = [int(label) for _, _, label in everything]

train_x, test_x, train_y, test_y = train_test_split(
    everything_x, everything_y,
    # set aside % for test
    test_size=.20,
    random_state=SEED,
    shuffle=False
)

def split_text(text_list: Sequence[Text]) -> Tuple[Sequence[Text], Sequence[Text]]:
    #split lines of text on [SNIPPET] and return lists of first and second excerpts
    texta = []
    textb = []
    for line in text_list:
        a, b = line.split("[SNIPPET]")
        texta.append(a)
        textb.append(b)
    return texta, textb


#split training data into two snippets
train_text_a, train_text_b = split_text(train_x)
test_text_a, test_text_b = split_text(test_x)
print("training data split!")


def stop_text(text_list: Sequence[Text]) -> Sequence[Text]:
    #take a list of strings and return a list of stop words for each string
    stop_words = []
    for row in text_list:
        row_stop = []
        for word in row.lower().split():
            if word in set(stopwords.words('english')):
                row_stop.append(word)
        stop_words.append(" ".join(row_stop))
    return stop_words

train_stop_a = stop_text(train_text_a)
train_stop_b = stop_text(train_text_b)
test_stop_a = stop_text(test_text_a)
test_stop_b = stop_text(test_text_b)
print("training stop words extracted!")


vectorizer = CountVectorizer(
  # case fold all text
  # before generating n-grams
  lowercase=False,
  # optionally apply the specified function
  # before counting n-grams
  preprocessor=None,
  # optionally provide a list of tokens to remove/ignore before generating n-grams
  stop_words=None,
  # specify a range of n-grams as (min_n, max_n)
  ngram_range=(1, 4),
  # "word", "char" (character), or "char_wb" n-grams
  analyzer="word",
  # whether or not to use binary counts
  binary=False
)


stop_vocab_vectors = vectorizer.fit(train_stop_a+train_stop_b)
Xa = vectorizer.transform(train_stop_a)
Xb = vectorizer.transform(train_stop_b)
X = abs(Xa-Xb)
print(f"length of the vectorizer vocab:{len(vectorizer.vocabulary_)}")
vocab_dim = len(vectorizer.vocabulary_)
# #
X_train = torch.tensor(X.toarray()).float()
y_train = torch.tensor(train_y).float()

class LogisticRegression(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()
        # xW^T + b
        self.linear = nn.Linear(in_features=input_size, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        return self.sigmoid(z)

# for reproducibility
torch.manual_seed(42)
# each datapoint in our X has two features
model = LogisticRegression(input_size=vocab_dim)


EPOCHS = 300
# use binary cross entropy as loss function
loss_fn = torch.nn.BCELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.0001
)

# put our model into training mode (i.e., tell it to calculate gradients)
model.train()
loss_tracker = []
# print(list(model.parameters()))
for i in range(1, EPOCHS + 1):
    # IMPORTANT: clear out our gradients from the last epoch.
    model.zero_grad()
    y_pred = model(X_train)
    y_true = y_train
    loss = loss_fn(y_pred.squeeze(), y_true)
    print(f"Epoch {i} loss: {loss}")
    # backward pass
    loss.backward()
    # actually update our parameters
    optimizer.step()
    loss_tracker.append((i, loss.item()))

test_a = vectorizer.transform(test_stop_a)
test_b = vectorizer.transform(test_stop_b)
Y = abs(test_a-test_b)
# #
X_test = torch.tensor(Y.toarray()).float()
y_test = torch.tensor(test_y).float()


# convert continuous-valued predictions into a 1 or 0.
binarize = lambda preds: (preds > 0.33).long()

# don't calculate gradients when evaluating
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_true = y_test.long()
    # pred_df = pd.DataFrame({'y_true':y_true,'y_pred':binarize(y_pred).flatten()})
    # print(pred_df)


t_total = torch.nansum(binarize(y_pred).flatten())
print(f"predicted total ones: {t_total}") #see how many 1s are predicted on held-out data
print(f"true total ones: {torch.nansum(y_true)}") #how many true 1s are in held-out data


##############################################################################
from sklearn.metrics import classification_report

y_hat = binarize(y_pred).flatten()
report = classification_report(y_true=y_true, y_pred=y_hat)

print(report)
