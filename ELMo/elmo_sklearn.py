import pandas as pd
import numpy as np
# ELMo
from allennlp.commands.elmo import ElmoEmbedder
elmo = None # global ELMo model
# Scikit Learn
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression

from enum import Enum
class Embedding(Enum):
    NormalizedAdding = 0  # divide the sum of embeddings with sentence length

class RegressionMethod(Enum):
    SVM_Regression = 0
    LinearRegression = 1

regression_model = {
    RegressionMethod.SVM_Regression: LinearSVR(epsilon=0.0, tol=1e-4, C=1.0),
    RegressionMethod.LinearRegression: LinearRegression(fit_intercept=True, normalize=True)
}


def PreprocessedDataLoader(train_path: str = "../preprocessed_data/train_preprocessed.csv",
                           test_path: str = "../preprocessed_data/test_preprocessed.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_X = train_df['comment_text'].astype(str)
    train_y = train_df['target'].values

    test_X = test_df['comment_text'].astype(str)

    return train_X, train_y, test_X, test_df

def SentenceEmbedding(sentence: str, embedding_method: int):
    # assume the sentence has been preprocessed
    tokens = sentence.split()
    vectors = elmo.embed_sentence(tokens)
    embeddings = vectors[2]  # last layer of the model output

    returnEmbedding = np.zeros((np.shape(embeddings)[1], ))
    
    if embedding_method == Embedding.NormalizedAdding:
        for word in embeddings:
            returnEmbedding += word
        returnEmbedding /= len(embeddings)

    return returnEmbedding

def AllToEmbedding(X, embedding_method: int):
    embedding = []
    for sentence in X:
        embedding.append(SentenceEmbedding(sentence, embedding_method))
    return embedding
        
def Regression(train_X, train_y, regression_method: int, embedding_method: int):
    model = regression_model[regression_method]
    train_X_embedding = AllToEmbedding(train_X, embedding_method)
    model.fit(train_X_embedding, train_y)
    return model

def Submission(test_X, test_df, model, embedding_method: int, output_name: str):
    test_X_embedding = AllToEmbedding(test_X, embedding_method)
    predictions = model.predict(test_X_embedding)
    submission = pd.DataFrame.from_dict({
        'id': test_df.id,
        'prediction': predictions
    })
    submission.to_csv('submission_' + output_name + '.csv', index=False)

def main():
    # Setting
    regression_method = RegressionMethod.LinearRegression
    embedding_method = Embedding.NormalizedAdding
    output_name = str(embedding_method) + "_" + str(regression_method)

    print("Loading preprocessed data...")
    train_X, train_y, test_X, test_df = PreprocessedDataLoader()
    print("Ready to train the regression model...")
    model = Regression(train_X, train_y, regression_method, embedding_method)
    print("Model trained, output submission...")
    Submission(test_X, test_df, model, embedding_method, output_name)

if __name__ == "__main__":
    print("Loading ELMo model...")
    elmo = ElmoEmbedder()
    main()
