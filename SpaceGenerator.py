import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import re

class SpaceGenerator:
  def __init__(self, model, vocabulary = [-1]):
    self.past_capacity = 5
    self.future_capacity = 5
    self.num_features = self.past_capacity + self.future_capacity + 1 # 1 for letter
    self.vocabulary = vocabulary
    self.model = model


  def fix_text(self, text, space_corrector = None):
    '''
    Fix the text given text and model
    '''
    text = re.sub(r'[^a-zA-Z.,:\'?;!]', '', text).replace(' ','')
    W = np.array(self.to_bytes_list(text))
    real_test_X = np.concatenate((self.sliding_window_past(W,window_size=self.past_capacity),W.reshape(-1,1),self.sliding_window_future(W,window_size=self.future_capacity)), axis= 1)
    real_test_X = np.stack(SpaceGenOneHotVectorizer.X_to_one_hot_matrix(real_test_X, vocabulary = self.vocabulary, max_len =self.num_features))
    new_data_reshaped = real_test_X.reshape((real_test_X.shape[0], 1, self.num_features*len(self.vocabulary)))
    predictions = self.model.predict(new_data_reshaped,verbose=0)
    D = [self.prob_to_decision(a) for a in predictions]
    fixed = self.to_correct(W, D)
    return fixed.strip()

  @staticmethod
  def create_decision_vector(W: list, C: list):
    '''
    Returns the Decision Vector(D),
    given Wrong Vector(W) and Correct Vector(C)
    '''
    D = []
    w_i = 0
    c_i = 0
    while w_i < len(W):
      if W[w_i] == C[c_i]:
          D.append('K')
          w_i += 1
          c_i += 1
      elif W[w_i] == 32 and C[c_i] != 32 :
          D.append('D')
          w_i += 1
      elif C[c_i] == 32 and W[w_i] != 32:
          D.append('I')
          c_i += 1
          w_i += 1
      else:
          c_i += 1
    return D


  @staticmethod
  def to_correct(W, D):
      '''
      Returns the correct text,
      given Wrong Vector(W) and Decision Vector(D)
      '''
      output_vec = []
      for i in range(0, len(D)):
        if D[i] == 'K':
          output_vec.append(W[i])
        elif D[i] == 'I':
          output_vec.append(32)
          output_vec.append(W[i])
        elif D[i] == 'D':
          pass
      decoded_text = bytes(output_vec).decode()
      return decoded_text


  @staticmethod
  def to_bytes_list(text: str, encoding = 'UTF-8'):
      '''
      Returns the bytes list of a given text
      '''
      return [b for b in bytes(text, encoding)]


  @staticmethod
  def to_one_hot_df(wrong_txt, D):
    '''
    Returns the one hot encoded dataframe,
    given Wrong Vector(W) and Decision Vector(D)
    '''
    df = pd.DataFrame({'letter':[l for l in wrong_txt],'decision':D})
    encoding =  OneHotEncoder()
    y_matrix =  encoding.fit_transform(df[['decision']])
    onehot_df = pd.DataFrame(y_matrix.toarray(), columns = encoding.get_feature_names_out(['decision']) )
    onehot_df = onehot_df.astype('int')
    example_df = pd.concat([df, onehot_df], axis=1)
    example_df =example_df.drop(['decision'], axis=1)
    return example_df


  @staticmethod
  def decode_vec(arr):
    '''
    Returns the decoded text,
    given the bytes list
    '''
    return bytes(arr).decode()


  @staticmethod
  def sliding_window_past(arr, window_size = 3):
    '''
    Returns the past sliding window of the given array and window size
    '''
    arr = list(arr)
    new_arr = []
    for i in range(len(arr)):
      start_window = max(0, i- window_size)
      tmp_seq = arr[start_window:i]
      if window_size - len(tmp_seq) ==0:
        new_arr.append(tmp_seq)
      else:
        new_arr.append([-1] * (window_size - len(tmp_seq)) + tmp_seq)
    return new_arr


  @staticmethod
  def sliding_window_future(arr, window_size = 3):
    '''
    Returns the future sliding window of the given array and window size
    '''
    arr = list(arr)
    seq = []
    for i in range(len(arr)):
      p = arr[i:i+window_size]
      if window_size - len(p) ==0:
        seq.append(p)
      else:
        seq.append(p + [-1] * (window_size - len(p)))
    return seq


  @staticmethod
  def insert_random_spaces(text, percent = .25):
    '''
    Returns the text with random spaces inserted
    '''
    l = list(text)
    rand_indices = np.random.randint(0, len(l)+1, int(np.round(len(l) * percent)))
    print(rand_indices)
    t = 1
    for i in range(len(l)+1):
      if i in rand_indices:
          l.insert(i + t, ' ')
          t+=1
    new_txt = ''.join(l).strip()
    return new_txt


  @staticmethod
  def prob_to_decision(a):
    '''
    Return I or K given probability vector
    '''
    if a[0] > a[1]:
      return 'I'
    else:
      return 'K'


class SpaceGenOneHotVectorizer:
    @staticmethod
    def string_vectorizer(string, vocabulary, max_len):
      empty = SpaceGenOneHotVectorizer.empty_matrix(max_len, vocabulary)
      empty[0, vocabulary.index(string)] = 1
      return empty

    @staticmethod
    def create_vocabulary(list_of_words):
      vocabulary = set(''.join(list_of_words))
      vocabulary = sorted([i for i in vocabulary])
      return vocabulary

    @staticmethod
    def empty_matrix(max_len, vocabulary):
      array = []
      for i in range(max_len):
        array.append([0]*len(vocabulary))
      return np.array(array)

    def X_to_one_hot_matrix(X, vocabulary, max_len):
        '''
        For a given X with past and future and vocaulary, returns a matrix representation for each sequence (row)
        '''
        df = pd.DataFrame(X)
        df_mat = df.applymap(lambda x: SpaceGenOneHotVectorizer.string_vectorizer(x, vocabulary = vocabulary, max_len = 1))
        df_mat['matrix_representation'] = df_mat.apply(lambda x: list(x), axis=1)
        df_mat['matrix_representation'] = df_mat['matrix_representation'].apply(lambda x: np.stack(x))
        df_mat['matrix_representation'] = df_mat['matrix_representation'].apply(lambda x: np.vstack(x))
        return df_mat['matrix_representation']
