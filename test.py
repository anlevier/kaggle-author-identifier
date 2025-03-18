from training import *
import csv

def read_test_csvtagged(csvtagged_path: str):
    """
    Reads rows from test csv .tagged file.
    Each row consists of 2 columns of information:

    COLUMN	DESCRIPTION
    ID	Unique ID for this datapoint
    TEXT	Two snippets of text separated by [SNIPPET]
    """
    rows = []
    with open(csvtagged_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append((row[0], row[1]))
    return rows

test_data = read_test_csvtagged('data/test.csv')[1:]

test_ids = [num for num,_ in test_data] #save ids to pair with test results later
test_X = [text for _,text in test_data]


test_text_A, test_text_B = split_text(test_X)

test_stop_A = stop_text(test_text_A)
test_stop_B = stop_text(test_text_B)
print("test stop words extracted!")

####Prepare test data for classification
test_A = vectorizer.transform(test_stop_A)
test_B = vectorizer.transform(test_stop_B)
Y = abs(test_A-test_B)

test_X = torch.tensor(Y.toarray()).float()

# convert continuous-valued predictions into a 1 or 0.
binarize = lambda preds: (preds > 0.33).long()

# don't calculate gradients when evaluating
model.eval()
with torch.no_grad():
    pred_Y = model(test_X)

####WRITE RESULTS TO CSV FILE###############################################

Y = binarize(pred_Y).flatten()


output = Y.tolist()
#confirm lengths are the same
#print(f"sum(output):{sum(output)}") #see how many 1s were predicted


res = []
for i in range(len(test_ids)):
    res.append((test_ids[i], output[i]))

with open('data/results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ID', 'LABEL'])
    for row in res:
        writer.writerow(row)

print("results written to file!")