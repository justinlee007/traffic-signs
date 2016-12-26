from sklearn.model_selection import train_test_split


def split_data_sets(train_features, train_labels):
    # Get randomized datasets for training and validation
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features,
        train_labels,
        test_size=0.05,
        random_state=832289)
    return train_features, valid_features, train_labels, valid_labels
