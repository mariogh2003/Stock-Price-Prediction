import pandas as pd
from ast import literal_eval
from embedding import embedding_main
import os

'''Define file paths'''
full_dataset = "C:/Users/mario/Downloads/Dataset.csv"
train_file_path = "C:/Users/mario/Downloads/training_dataset.csv"
test_file_path = "C:/Users/mario/Downloads/testing_dataset.csv"
combined_news_path = "C:/Users/mario/Downloads/Combined_News_DJIA.csv"
embedding_csv = "C:/Users/mario/Downloads/Word_embedding.csv"
merged_csv = "C:/Users/mario/Downloads/Merged_dataset.csv"
folder_path = "C:/Users/mario/Downloads/archive/Stocks"

'''Call the main function of the embedding script to generate embeddings'''
embedding_main(combined_news_path, embedding_csv)

def txt_to_csv(txt_file_path, csv_file_path, delimiter=' '):
    '''Convert txt file to csv'''
    df = pd.read_csv(txt_file_path, delimiter=delimiter)
    df.to_csv(csv_file_path, index=False)
    return csv_file_path

def train_and_test(csv_path, train_path, test_path):
    '''Separate dataset into train and test'''
    new_data = pd.read_csv(csv_path)
    train_num = int(new_data.shape[0] * 4 / 5)
    
    new_train = new_data[:train_num]
    new_test = new_data[train_num:]

    existing_train = pd.DataFrame()
    existing_test = pd.DataFrame()
    
    try: 
        existing_train = pd.read_csv(train_path)
        existing_test = pd.read_csv(test_path)
    except pd.errors.EmptyDataError:
        print('\nFile is empty')
        
    train = pd.concat([existing_train, new_train])
    test = pd.concat([existing_test, new_test])

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    return pd.read_csv(train_path), pd.read_csv(test_path)

def split_column_in_excel(df, output_path):
    '''Split csv format into multiple columns and filter the date to the available news dates'''
    df_expanded = df[df.columns[0]].str.split(',', expand=True)
    df_expanded.columns = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
    df_expanded['Date'] = pd.to_datetime(df_expanded['Date'])

    df_filtered = df_expanded[(df_expanded['Date'] >= '2008-08-08') & (df_expanded['Date'] <= '2016-07-01')].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered['Open'] = df_filtered['Open'].astype(float)
    df_filtered['pct_change'] = df_filtered['Open'].pct_change()
    df_filtered.loc[0, 'pct_change'] = 0
    
    df_filtered['Date_diff'] = df_filtered['Date'].diff().abs().dt.days
    condition = df_filtered['Date_diff'] > 10
    indices = df_filtered.index[condition]

    for idx in indices:
        if idx < df_filtered.shape[0]:
            df_filtered.loc[idx, 'pct_change'] = 0

    df_filtered = df_filtered.drop(["High", "Low", "OpenInt"], axis=1) 
    df_filtered.to_csv(output_path, index=False)

    return df_filtered

def final_csv():
    '''Take how many files you want and add them in training and testing dataset'''
    open(train_file_path, 'w').close()
    open(test_file_path, 'w').close()

    files = os.listdir(folder_path)

    for i in range(10):
        file_path = os.path.join(folder_path, files[i])
        csv_file = txt_to_csv(file_path, full_dataset, delimiter=' ')
        train_df, test_df = train_and_test(csv_file, train_file_path, test_file_path)
    
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    train_df = split_column_in_excel(train_df, train_file_path)
    test_df = split_column_in_excel(test_df, test_file_path)

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)
    
    print(f"\nTrain data has been saved to '{train_file_path}'.")
    print(f"\nTest data has been saved to '{test_file_path}'.")

def merge_datasets(embedding_file_path, data_df, output_file_path):
    '''Combine two datasets based on their shared dates found in the Date column'''
    embeddings_df = pd.read_csv(embedding_file_path)
    
    embeddings_df['Date'] = pd.to_datetime(embeddings_df['Date'])
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    
    merged_df = pd.merge(data_df, embeddings_df, on='Date', how='left')
    
    merged_df.to_csv(output_file_path, index=False)
    print(f"\nMerged data has been saved to '{output_file_path}'.")

def convert_list_strings_to_columns(df, column_name):
    '''Convert list strings to individual columns
       Fill NaN and empty values with an empty list'''
    df[column_name] = df[column_name].apply(lambda x: '[]' if pd.isna(x) or x == '' else x)
    list_cols = df[column_name].apply(literal_eval)
    list_df = pd.DataFrame(list_cols.tolist(), index=df.index)
    list_df.columns = [f'{column_name}_{i}' for i in list_df.columns]
    df = df.drop(column_name, axis=1)
    df = pd.concat([df, list_df], axis=1)
    return df

def dummy_date(df):
    '''Extract day and merge embeddings'''
    merge_datasets(embedding_csv, df, merged_csv)
    df = pd.read_csv(merged_csv)
    df["day"] = df["Date"].apply(lambda x: pd.to_datetime(x).day)
    return df

def pre_processing(df):
    '''Convert to % and drop unnecessary parameters'''
    df = dummy_date(df)
    df = df.drop(["Date"], axis=1)

    embedding_columns = [col for col in df.columns if col.endswith('_embedding')]
    for column_name in embedding_columns:
        df = convert_list_strings_to_columns(df, column_name)
    
    df.to_csv(merged_csv, index=False)
    return df
