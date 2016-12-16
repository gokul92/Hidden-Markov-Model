import csv
import numpy as np

class get_data_id(object):
    def __init__(self, filename, sgx_nifty):
        self.file = open(filename, "r")
        self.csv_output = csv.reader(self.file)
        self.sgx = sgx_nifty

        self.nifty_str = []
        for row in self.csv_output:
            self.nifty_str.append(row)

        self.file.close()

        self.csv_head = self.nifty_str[0]
        if self.sgx == 0:
            self.csv_head.pop()

        self.nifty_str = self.nifty_str[1:]

        if self.sgx == 0:
            for row in self.nifty_str:
                row.pop()

        if self.sgx != 0:
            for row in self.nifty_str:
                newstr = row[1].replace(":", "")
                row[1] = newstr

        self.nifty = np.array([[float(i) for i in row] for row in self.nifty_str])

    def get_datelist(self):
        self.date = np.unique(self.nifty[:, 0])
        return self.date

    def get_open_price(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        self.open_price = self.nifty[self.indices_list[0], 2]
        return self.open_price

    def get_high_price(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        self.high_price = np.amax(self.nifty[self.indices_list[0]:self.indices_list[-1], 3])
        return self.high_price

    def get_low_price(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        self.low_price = np.amin(self.nifty[self.indices_list[0]:self.indices_list[-1], 4])
        return self.low_price

    def get_close_price(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        self.close_price = self.nifty[self.indices_list[-1], 5]
        return self.close_price

    def get_min_ohlc(self, date, time):
        loc = int(np.where((date == self.nifty[:, 0]) & (time == self.nifty[:, 1]))[0])
        return self.nifty[loc, 2:6]

    def get_min_open_pricelist(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        return np.array(self.nifty[self.indices_list[0]:self.indices_list[-1], 2])

    def get_min_high_pricelist(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        return np.array(self.nifty[self.indices_list[0]:self.indices_list[-1], 3])

    def get_min_low_pricelist(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        return np.array(self.nifty[self.indices_list[0]:self.indices_list[-1], 4])

    def get_min_close_pricelist(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        return np.array(self.nifty[self.indices_list[0]:self.indices_list[-1], 5])

    def get_min_volume_pricelist(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        return np.array(self.nifty[self.indices_list[0]:self.indices_list[-1], 6])

    def get_timelist(self, date):
        self.indices_list = np.where(date == self.nifty[:, 0])[0]
        return np.array(self.nifty[self.indices_list[0]:self.indices_list[-1], 1])

    def get_head(self):
        return self.csv_head
