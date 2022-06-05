#only applicable when train.csv and test.csv is not readily available.

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)