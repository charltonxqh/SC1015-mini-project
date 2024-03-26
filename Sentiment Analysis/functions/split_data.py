from sklearn.model_selection import train_test_split

def split_data(data):
  train_valid_data, test_data = train_test_split(data, test_size=0.5)

  train_data, valid_data = train_test_split(train_valid_data, test_size=0.25)

  return train_data, valid_data, test_data