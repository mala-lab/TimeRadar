import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset
from utils.tools import DataPool
import warnings
import h5py

warnings.filterwarnings('ignore')


class TrainSegLoader(Dataset):
    def __init__(self, data_name, data_path, train_length, seq_len, pred_len, stride, flag="train", percentage=0.1, discrete_channels=None):
        self.data_name = data_name
        self.data_path = data_path
        self.train_length = train_length
        self.flag = flag
        self.stride = stride
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.percentage = percentage
        self.discrete_channels = discrete_channels
        
        # 1.read data
        self.__read_data__()

    def __read_data__(self, nrows=None):
        data = pd.read_csv(self.data_path)
        label_exists = "label" in data["cols"].values
        all_points = data.shape[0]
        columns = data.columns
        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()
        is_univariate = n_points == all_points
        n_cols = all_points // n_points
        df = pd.DataFrame()
        cols_name = data["cols"].unique()
        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            df[cols_name[0]] = data.iloc[:, 0]
        if label_exists:
            last_col_name = df.columns[-1]
            df.rename(columns={last_col_name: "label"}, inplace=True)
        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]

        # if 'synthetic' in self.data_path:
        # train_data = df.iloc[:int(self.train_length), :].to_numpy()
        # train_label = train_data
        # # 3.test
        # test_data = df.iloc[int(self.train_length):, :].to_numpy()
        # test_label =  test_data
        # else:
        # 2.train
        train_data = df.iloc[:int(self.train_length), :]
        train_data, train_label =  (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )
        # 3.test
        test_data = df.iloc[int(self.train_length):, :]
        test_data, test_label =  (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )

        # 4.process
        if self.discrete_channels is not None:
            train_data = np.delete(train_data, self.discrete_channels, axis=-1)
            test_data = np.delete(test_data, self.discrete_channels, axis=-1)
        
        # if 'synthetic' not in self.data_path:
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if self.flag == "init":
            self.init = train_data
            self.init_label = train_label
            print(f"\nInit length: {self.init.shape[0]}")
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-self.percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label
        
            print(f"\nTrain, Val, Test lengths: {self.train.shape[0]}, {self.val.shape[0]}, {self.test.shape[0]}")
    

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.seq_len) // self.stride + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.seq_len) // self.stride + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.seq_len) // self.stride + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.seq_len) // self.stride + 1
        else:
            return (self.test.shape[0] - self.seq_len) // self.seq_len + 1

    def __getitem__(self, index, eps=1):
        index = index * self.stride
        if self.flag == "train":       
            return np.float32(self.train[index: index + self.seq_len]), np.float32(self.train_label[index: index + self.seq_len])
        elif self.flag == "val":
            return np.float32(self.val[index: index + self.seq_len]), np.float32(self.val_label[index: index + self.seq_len])
        elif self.flag == "test":
            return np.float32(self.test[index: index + self.seq_len]), np.float32(self.test_label[index: index + self.seq_len])
        elif self.flag == "init":
            return np.float32(self.init[index: index + self.seq_len]), np.float32(self.init_label[index: index + self.seq_len])
        else:
            return np.float32(self.test[index // self.stride * self.seq_len: index// self.stride * self.seq_len+ self.seq_len]), np.float32(self.test_label[index // self.stride * self.seq_len: index // self.stride * self.seq_len + self.seq_len])




class PreTrainSegLoader(Dataset):
    def __init__(self, data_path, seq_len, pred_len, stride, flag="train", split=0.8):
        # init
        assert flag in ['train', 'val', 'test', 'init']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'init': 3}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.flag = flag
        self.stride = stride     # default stride 1
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.split = split
        # self.shuffle = shuffle
        self.data_list = []
        self.label_list = []
        self.n_window_list = []

        # TAB: index column name of the metadata
        self._INDEX_COL = "file_name"
        # TAB: name of the data folder
        self._DATA_FOLDER_NAME = "data"
        # # TAB: name of the covariates folder
        # self._COVARIATES_FOLDER_NAME = "covariates"
        # TAB: index column name of the metadata
        self._INDEX_COL = "file_name"

        # 1.read data
        self.__read_data__()

    
    def update_meta_index(self) -> pd.DataFrame:
        """
        Check if there are any user-added dataset files in the dataset folder

        Attempt to register them in the metadata and load metadata from a local csv file

        :return: metadata
        :rtype: pd.DataFrame
        """

        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def is_st(self, data: pd.DataFrame) -> bool:
        """
        Checks if data of the CSV file are in spatial-temporal format.

        :param data: The series data.
        :return: Are all values in 'cols' column are in spatial-temporal format.
        """
        return data.shape[1] == 4

    def process_data_df(self, data: pd.DataFrame, nrows=None) -> pd.DataFrame:
        """
        Read the data file and return DataFrame.

        According to the provided file path, read the data file and return the corresponding DataFrame.

        :param data: Data frame to read.
        :return: The DataFrame of the content of the data file.
        """
        label_exists = "label" in data["cols"].values

        all_points = data.shape[0]

        columns = data.columns

        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()

        is_univariate = n_points == all_points

        n_cols = all_points // n_points
        df = pd.DataFrame()

        cols_name = data["cols"].unique()

        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]

            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        else:
            df[cols_name[0]] = data.iloc[:, 0]

        if label_exists:
            # Get the column name of the last column
            last_col_name = df.columns[-1]
            # Renaming the last column as "label"
            df.rename(columns={last_col_name: "label"}, inplace=True)

        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]
        
        return df


    def process_data_np(self, df: pd.DataFrame, nrows=None) -> np.ndarray:
        """
        Convert spatial-temporal data from a DataFrame

        to a three-dimensional(time stamp,feature,sensor)  numpy array.

        :param df: Spatial-temporal data.
        :param nrows: Optional, number of rows to retain. Default is None, retaining all rows.
        :return: Three-dimensional(time stamp,feature,sensor) numpy array of the spatial temporal data.
        """
        pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

        sensors = df["id"].unique()
        features = df["cols"].unique()
        pivot_df = pivot_df.reindex(
            columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
        )

        data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
        data_np = np.transpose(data_np, (0, 2, 1))

        if nrows is not None:
            data_np = data_np[:nrows, :, :]

        return data_np


    def __read_data__(self):
        # monash
        monash_dataset_path = os.path.join(self.data_path, 'Monash+')
        for root, dirs, files in os.walk(monash_dataset_path):
            dirs.sort()
            files.sort()
            for file in files:
                subdata_path = os.path.join(root, file)
                with h5py.File(subdata_path, 'r') as input_hf:
                    data = input_hf['data']
                    series = data[:, 0]
                    labels = data[:, 1]

                    num_train = int(series.shape[0] * self.split)
                    num_test = int(series.shape[0] * (1 - self.split) / 2)
                    num_vali = series.shape[0] - num_train - num_test

                    border1s = [0, num_train, series.shape[0] - num_test, 0]
                    border2s = [num_train, num_train + num_vali, series.shape[0], num_train + num_vali]   # init for spot

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]

                    data = series[border1:border2]
                    label = labels[border1:border2]
                    n_window = data.shape[0]
                    if n_window <= 0:
                        continue
                    n_var = data.shape[2]
                    self.data_list.append(data)
                    self.label_list.append(label)
                    all_n_window = n_window * n_var
                    self.n_window_list.append(all_n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + all_n_window)
        
        if len(self.n_window_list) == 0:
            raise ValueError(f"No samples found under {self.data_path}")
        
        # if self.shuffle:
        #     self._shuffle_once()
        print("\nTotal number of windows in merged dataset: ", self.n_window_list[-1])
        
    def __len__(self):
        return self.n_window_list[-1]

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        n_window = len(self.data_list[dataset_index])

        c_begin = index // n_window     # select variable
        s_begin = index % n_window      # select start window
        seq_x = np.float32(self.data_list[dataset_index][s_begin, :, c_begin:c_begin + 1])
        lab_y = np.float32(self.label_list[dataset_index][s_begin, :, c_begin:c_begin + 1])
        return seq_x, lab_y
    

# AddPre means add prediction process
class TrainSegLoaderAddPre(Dataset):
    def __init__(self, data_name, data_path, train_length, seq_len, pred_len, stride, flag="train", percentage=0.1, discrete_channels=None):
        self.data_name = data_name
        self.data_path = data_path
        self.train_length = train_length
        self.flag = flag
        self.stride = stride
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.percentage = percentage
        self.discrete_channels = discrete_channels
        # 1.read data
        self.__read_data__()
        
    def __read_data__(self, nrows=None):
        data = pd.read_csv(self.data_path)
        label_exists = "label" in data["cols"].values
        all_points = data.shape[0]
        columns = data.columns
        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()
        is_univariate = n_points == all_points
        n_cols = all_points // n_points
        df = pd.DataFrame()
        cols_name = data["cols"].unique()
        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            df[cols_name[0]] = data.iloc[:, 0]
        if label_exists:
            last_col_name = df.columns[-1]
            df.rename(columns={last_col_name: "label"}, inplace=True)
        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]

        # 2.train
        train_data = df.iloc[:self.train_length, :]
        train_data, train_label =  (
            train_data.loc[:, train_data.columns != "label"].to_numpy(),
            train_data.loc[:, ["label"]].to_numpy(),
        )

        # 3.test
        test_data = df.iloc[self.train_length:, :]
        test_data, test_label =  (
            test_data.loc[:, test_data.columns != "label"].to_numpy(),
            test_data.loc[:, ["label"]].to_numpy(),
        )

        if self.data_name == 'PSM':
            train_data = np.nan_to_num(train_data)
            test_data = np.nan_to_num(test_data)

        # 4.process
        if self.discrete_channels is not None:
            train_data = np.delete(train_data, self.discrete_channels, axis=-1)
            test_data = np.delete(test_data, self.discrete_channels, axis=-1)
        
        self.scaler = StandardScaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        if self.flag == "init":
            self.init = train_data
            self.init_label = train_label
            print(f"\nInit length: {self.init.shape[0]}")
        else:
            train_end = int(len(train_data) * 0.8)
            train_start = int(train_end*(1-self.percentage))
            self.train = train_data[train_start:train_end]
            self.train_label = train_label[train_start:train_end]
            self.val = train_data[train_end:]
            self.val_label = train_label[train_end:]
            self.test = test_data
            self.test_label = test_label
        
            print(f"\nTrain, Val, Test lengths: {self.train.shape[0]}, {self.val.shape[0]}, {self.test.shape[0]}")
        
        # For SEMPO, segment variates
        # self.n_var = train_data.shape[-1]
        # if self.flag == "train":
        #     self.n_timepoint = (self.train.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        # elif self.flag == "val":
        #     self.n_timepoint = (self.val.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        # elif self.flag == "test":
        #     self.n_timepoint = (self.test.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        # elif self.flag == "init":
        #     self.n_timepoint = (self.init.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        # else:
        #     self.n_timepoint = (self.test.shape[0] - self.seq_len - self.pred_len) // self.seq_len + 1


    def __len__(self):
        # For SEMPO, segment variates
        # return int(self.n_var * self.n_timepoint)
        if self.flag == "train":
            return (self.train.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        elif self.flag == "val":
            return (self.val.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        elif self.flag == "test":
            return (self.test.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        elif self.flag == "init":
            return (self.init.shape[0] - self.seq_len - self.pred_len) // self.stride + 1
        else:
            return (self.test.shape[0] - self.seq_len - self.pred_len) // self.seq_len + 1

    def __getitem__(self, index, eps=1):
        # For SEMPO, segment variates
        # c_begin = index // self.n_timepoint  # select variable
        # s_begin = index % self.n_timepoint   # select start timepoint
        # s_begin = self.stride * s_begin
        # s_end = s_begin + self.seq_len
        # r_begin = s_end
        # r_end = r_begin + self.pred_len

        # if self.flag == "train":       
        #     seq_x = np.float32(self.train[s_begin: s_end, c_begin: c_begin + 1])
        #     lab_x = np.float32(self.train_label[s_begin: s_end, 0: 1])
        #     seq_y = np.float32(self.train[r_begin: r_end, c_begin: c_begin + 1])
        #     lab_y = np.float32(self.train_label[r_begin: r_end, 0: 1])
        # elif self.flag == "val":
        #     seq_x = np.float32(self.val[s_begin: s_end, c_begin: c_begin + 1])
        #     lab_x = np.float32(self.val_label[s_begin: s_end, 0: 1])
        #     seq_y = np.float32(self.val[r_begin: r_end, c_begin: c_begin + 1])
        #     lab_y = np.float32(self.val_label[r_begin: r_end, 0: 1])
        # elif self.flag == "test":
        #     seq_x = np.float32(self.test[s_begin: s_end, c_begin: c_begin + 1])
        #     lab_x = np.float32(self.test_label[s_begin: s_end, 0: 1])
        #     seq_y = np.float32(self.test[r_begin: r_end, c_begin: c_begin + 1])
        #     lab_y = np.float32(self.test_label[r_begin: r_end, 0: 1])
        # elif self.flag == "init":
        #     seq_x = np.float32(self.init[s_begin: s_end, c_begin: c_begin + 1])
        #     lab_x = np.float32(self.init_label[s_begin: s_end, 0: 1])
        #     seq_y = np.float32(self.init[r_begin: r_end, c_begin: c_begin + 1])
        #     lab_y = np.float32(self.init_label[r_begin: r_end, 0: 1])
        # else:
        #     s_begin = s_begin // self.stride * self.seq_len
        #     s_end = s_begin + self.seq_len
        #     r_begin = s_end
        #     r_end = r_begin + self.pred_len
        #     seq_x = np.float32(self.test[s_begin: s_end, c_begin: c_begin + 1])
        #     lab_x = np.float32(self.test_label[s_begin: s_end, 0: 1])
        #     seq_y = np.float32(self.test[r_begin: r_end, c_begin: c_begin + 1])
        #     lab_y = np.float32(self.test_label[r_begin: r_end, 0: 1])
        # return seq_x, lab_x, seq_y, lab_y
        index = index * self.stride
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.flag == "train":       
            return np.float32(self.train[s_begin: s_end]), np.float32(self.train_label[s_begin: s_end]), np.float32(self.train[r_begin : r_end]), np.float32(self.train_label[r_begin : r_end])
        elif self.flag == "val":
            return np.float32(self.val[s_begin: s_end]), np.float32(self.val_label[s_begin: s_end]), np.float32(self.val[r_begin : r_end]), np.float32(self.val_label[r_begin : r_end])
        elif self.flag == "test":
            return np.float32(self.test[s_begin: s_end]), np.float32(self.test_label[s_begin: s_end]), np.float32(self.test[r_begin : r_end]), np.float32(self.test_label[r_begin : r_end])
        elif self.flag == "init":
            return np.float32(self.init[s_begin: s_end]), np.float32(self.init_label[s_begin: s_end]), np.float32(self.init[r_begin : r_end]), np.float32(self.init_label[r_begin : r_end])
        else:
            s_begin = index // self.stride * self.seq_len
            s_end = s_begin + self.seq_len
            r_begin = s_end
            r_end = r_begin + self.pred_len
            return np.float32(self.test[s_begin: s_end]), np.float32(self.test_label[s_begin: s_end]), np.float32(self.test[r_begin: r_end]), np.float32(self.test_label[r_begin: r_end])



class PreTrainSegLoaderAddPre(Dataset):
    def __init__(self, data_path, seq_len, pred_len, stride, flag="train", split=0.8):
        # init
        assert flag in ['train', 'val', 'test', 'init']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'init': 3}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.split = split
        self.data_list = []
        self.label_list = []
        self.n_window_list = []

        # TAB: index column name of the metadata
        self._INDEX_COL = "file_name"
        # TAB: name of the data folder
        self._DATA_FOLDER_NAME = "data"
        # # TAB: name of the covariates folder
        # self._COVARIATES_FOLDER_NAME = "covariates"
        # TAB: index column name of the metadata
        self._INDEX_COL = "file_name"

        # 1.read data
        self.__read_data__()

    def update_meta_index(self) -> pd.DataFrame:
        """
        Check if there are any user-added dataset files in the dataset folder

        Attempt to register them in the metadata and load metadata from a local csv file

        :return: metadata
        :rtype: pd.DataFrame
        """

        metadata = pd.read_csv(self.metadata_path)
        metadata.set_index(self._INDEX_COL, drop=False, inplace=True)
        return metadata

    def is_st(self, data: pd.DataFrame) -> bool:
        """
        Checks if data of the CSV file are in spatial-temporal format.

        :param data: The series data.
        :return: Are all values in 'cols' column are in spatial-temporal format.
        """
        return data.shape[1] == 4

    def process_data_df(self, data: pd.DataFrame, nrows=None) -> pd.DataFrame:
        """
        Read the data file and return DataFrame.

        According to the provided file path, read the data file and return the corresponding DataFrame.

        :param data: Data frame to read.
        :return:  The DataFrame of the content of the data file.
        """
        label_exists = "label" in data["cols"].values

        all_points = data.shape[0]

        columns = data.columns

        if columns[0] == "date":
            n_points = data.iloc[:, 2].value_counts().max()
        else:
            n_points = data.iloc[:, 1].value_counts().max()

        is_univariate = n_points == all_points

        n_cols = all_points // n_points
        df = pd.DataFrame()

        cols_name = data["cols"].unique()

        if columns[0] == "date" and not is_univariate:
            df["date"] = data.iloc[:n_points, 0]
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        elif columns[0] != "date" and not is_univariate:
            col_data = {
                cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
                for j in range(n_cols)
            }
            df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

        elif columns[0] == "date" and is_univariate:
            df["date"] = data.iloc[:, 0]
            df[cols_name[0]] = data.iloc[:, 1]

            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        else:
            df[cols_name[0]] = data.iloc[:, 0]

        if label_exists:
            # Get the column name of the last column
            last_col_name = df.columns[-1]
            # Renaming the last column as "label"
            df.rename(columns={last_col_name: "label"}, inplace=True)

        if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
            df = df.iloc[:nrows, :]

        return df


    def process_data_np(self, df: pd.DataFrame, nrows=None) -> np.ndarray:
        """
        Convert spatial-temporal data from a DataFrame

        to a three-dimensional(time stamp,feature,sensor)  numpy array.

        :param df: Spatial-temporal data.
        :param nrows: Optional, number of rows to retain. Default is None, retaining all rows.
        :return: Three-dimensional(time stamp,feature,sensor) numpy array of the spatial temporal data.
        """
        pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

        sensors = df["id"].unique()
        features = df["cols"].unique()
        pivot_df = pivot_df.reindex(
            columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
        )

        data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
        data_np = np.transpose(data_np, (0, 2, 1))

        if nrows is not None:
            data_np = data_np[:nrows, :, :]

        return data_np

    def __read_data__(self):
        # monash
        monash_dataset_path = os.path.join(self.data_path, 'Monash+')
        for root, dirs, files in os.walk(monash_dataset_path):
            dirs.sort()
            files.sort()
            for file in files:
                subdata_path = os.path.join(root, file)
                with h5py.File(subdata_path, 'r') as input_hf:
                    data = input_hf['data']
                    series = data[:, 0]
                    labels = data[:, 1]

                    num_train = int(series.shape[0] * self.split)
                    num_test = int(series.shape[0] * (1 - self.split) / 2)
                    num_vali = series.shape[0] - num_train - num_test

                    border1s = [0, num_train, series.shape[0] - num_test, 0]
                    border2s = [num_train, num_train + num_vali, series.shape[0], num_train + num_vali]   # init for spot

                    border1 = border1s[self.set_type]
                    border2 = border2s[self.set_type]
                    # (1168,100,1)
                    data = series[border1:border2]
                    label = labels[border1:border2]

                    # Scale has been performed when constructing the samples.
                    # train_data = data[border1s[0]:border2s[0]]
                    # self.scaler.fit(train_data)
                    # data = self.scaler.transform(data)

                    # Considering the horizon window, two strides are needed to obtain the horizon window.
                    n_window = len(data) - 2
                    if n_window <= 0:
                        continue
                    n_var = data.shape[2]
                    self.data_list.append(data)
                    self.label_list.append(label)
                    all_n_window = n_window * n_var
                    self.n_window_list.append(all_n_window if len(
                        self.n_window_list) == 0 else self.n_window_list[-1] + all_n_window)
            
        print("\nTotal number of windows in merged dataset: ", self.n_window_list[-1])
        
    def __len__(self):
        return self.n_window_list[-1]

    def __getitem__(self, index):
        assert index >= 0
        # find the location of one dataset by the index
        dataset_index = 0
        while index >= self.n_window_list[dataset_index]:
            dataset_index += 1

        index = index - self.n_window_list[dataset_index - 1] if dataset_index > 0 else index
        # consider forecasting
        n_window = len(self.data_list[dataset_index]) - 2

        c_begin = index // n_window   # select variable
        s_begin = index % n_window   # select start window
        r_begin = s_begin + 2
        seq_x = np.float32(self.data_list[dataset_index][s_begin, :, c_begin:c_begin + 1])
        lab_x = np.float32(self.label_list[dataset_index][s_begin, :, c_begin:c_begin + 1])
        seq_y = np.float32(self.data_list[dataset_index][r_begin, :, c_begin:c_begin + 1])
        lab_y = np.float32(self.label_list[dataset_index][r_begin, :, c_begin:c_begin + 1])
        return seq_x, lab_x, seq_y, lab_y