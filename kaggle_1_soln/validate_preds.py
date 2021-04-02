import pandas as pd
import numpy as np

preds = pd.read_csv("kaggle_1_soln/test_preds.csv")

bird_names = pd.unique(preds["target"])

def top5_acc(df):
    predictions = df.preds.values
    targets = df.target.values

    acc =0.0
    for p,t in zip(predictions,targets):
        if t in p:
            acc+=1
    #print("Top 5 Accuracy: ",acc/len(predictions))
    return acc/len(predictions)

for bird_name in bird_names:
    req = preds[preds["target"]==bird_name]
    #accuracy = np.sum(req["preds"]==req["target"].values)/len(req)
    accuracy = top5_acc(req)
    print("BIRD NAME: ",bird_name,"TOP 3 ACCURACY: ",accuracy)
