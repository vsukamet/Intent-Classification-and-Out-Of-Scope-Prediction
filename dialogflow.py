from __future__ import print_function
import pandas as pd
import os
import uuid
import datetime
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import dialogflow_v2 as dialogflow
import gcsfs
from sklearn.metrics import f1_score, precision_score, recall_score
from google.auth.transport.requests import Request

df = pd.read_csv('testdataoos.csv')
print('Dataframe size = ', df.shape)
print(df.head(10))

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'testing-agent-new-xhkyjk-9088a6f045df.json'
# Replace it with your Service Account JSON file
PROJECT_ID = 'testing-agent-new-xhkyjk'
SESSION_ID = str(uuid.uuid4())
LANGUAGE_CODE = 'en'
now = datetime.datetime.now()





def detect_intent_texts(project_id, session_id, texts, language_code):
    import dialogflow_v2 as dialogflow
    session_client = dialogflow.SessionsClient()
    print(session_client)
    session = session_client.session_path(project_id, session_id)
    print(session)
    intents = []
    for text in texts:
        text_input = dialogflow.types.TextInput(text=text, language_code=language_code)
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
        intents.append(response.query_result.intent.display_name)
    return intents


texts = df['Query']
col_name = 'Actual Outcome {}/{}'.format(now.month, now.day)
# print(col_name)
df[col_name] = detect_intent_texts(PROJECT_ID, SESSION_ID, texts, LANGUAGE_CODE)
# print(df.head(10))


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def confusion_matrix_plot(y_true, y_pred, labels, label_maps=None, figsize=(10, 10), vmin=None, vmax=None,title=None):
    if label_maps is not None:
        y_true = [label_maps[k] for k in y_true]
        y_pred = [label_maps[k] for k in y_pred]
        labels = [label_maps[k] for k in labels]

    cm_count = confusion_matrix(y_true, y_pred, labels)
    nrow, ncol = cm_count.shape
    annot = np.zeros((nrow, ncol)).astype(str)
    row_sum = np.sum(cm_count, axis=1, keepdims=True)
    # print(row_sum)
    # print(cm_count)
    cm_perc = cm_count / row_sum


    for i in range(nrow):
        for j in range(ncol):
            c = cm_count[i, j]
            p = cm_perc[i, j]
            s = row_sum[i][0]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '{:.1%}\n{}/{}'.format(p, c, s)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_count, vmin=vmin, vmax=vmax,
                cmap=sns.cm.rocket_r,
                annot=annot, fmt='',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax)
    plt.title(title)
    plt.xlabel('Actual Outcome')
    plt.ylabel('Expected Outcome')
    plt.xticks(rotation=45)
    plt.show()
    fig.show()


y_true = df['IntentName']
y_pred = df[col_name]
print(y_pred)
labels = np.unique(y_true)
print(labels)
confusion_matrix_plot(y_true, y_pred, labels, figsize=(50, 50), vmin=0, vmax=10,title='Confusion Matrix with All Test Intents')

print('___Results on {}/{}___'.format(now.month, now.day))
print('Marco F1 Score: ', round(f1_score(y_true, y_pred, average='macro'), 2))
print('Precision Score: ', round(precision_score(y_true, y_pred, average='macro', zero_division=1), 2))
print('Recall Score: ', round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 2))


