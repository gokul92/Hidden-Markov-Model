import csv
import numpy as np

class get_data_eod(object):
    def __init__(self, filename, sgx_nifty, date=0, op=0, high=0, low=0, close=0, volume=0):
        self.file = open(filename, "r")
        self.csv_output = csv.reader(self.file)
        self.sgx = sgx_nifty

        self.nifty_str = []
        for row in self.csv_output:
            self.nifty_str.append(row)

        self.file.close()

        self.csv_head = self.nifty_str[0]
        self.csv_head.pop()

        self.nifty_str = self.nifty_str[1:]

        for row in self.nifty_str:
            row.pop()

        if self.sgx != 0:
            for row in self.nifty_str:
                newstr = row[0].replace("-", "")
                row[0] = newstr

        self.nifty = np.array([[float(i) for i in row] for row in self.nifty_str])

        if self.sgx != 0:
            self.nifty = self.nifty[np.argsort(self.nifty[:, 0])]

        self.date = date
        self.open_price = op
        self.high_price = high
        self.low_price = low
        self.close_price = close
        self.volume = volume

    def get_date(self):
        self.date = np.array(self.nifty[:, 0])
        return self.date

    def get_open_price(self):
        if self.sgx == 0:
            self.open_price = np.array(self.nifty[:, 2])
        else:
            self.open_price = np.array(self.nifty[:, 1])
        return self.open_price

    def get_high_price(self):
        if self.sgx == 0:
            self.high_price = np.array(self.nifty[:, 3])
        else:
            self.high_price = np.array(self.nifty[:, 2])
        return self.high_price

    def get_low_price(self):
        if self.sgx == 0:
            self.low_price = np.array(self.nifty[:, 4])
        else:
            self.low_price = np.array(self.nifty[:, 3])
        return self.low_price

    def get_close_price(self):
        if self.sgx == 0:
            self.close_price = np.array(self.nifty[:, 5])
        else:
            self.close_price = np.array(self.nifty[:, 4])
        return self.close_price

    def get_volume(self):
        self.volume = np.array(self.nifty[:, 6])
        return self.volume

    def get_head(self):
        return self.csv_head
