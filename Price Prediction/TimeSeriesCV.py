
class TimeSeriesCV:
    """
    Generates train_idx, test_idx tuples
    assumes the dataframe has index 'time'
    """
    def __init__(self, n_cv, train_length, test_length, lookahead):
        self.n_cv = n_cv
        self.train_length = train_length
        self.test_length = test_length
        self.lookahead = lookahead

    def split(self, data):
        times = data.index.unique()
        time_reversed = sorted(times, reverse=True)

        split_idx = []
        for i in range(self.n_cv):
            test_end = i * self.test_length
            test_start = test_end + self.test_length
            train_end = test_start + self.lookahead - 1
            train_start = train_end  + self.train_length + self.lookahead - 1
            split_idx.append([train_start, train_end, test_start, test_end])

        time_stmp = data.reset_index()[['time']]
        for train_start, train_end, test_start, test_end in split_idx:
            train_idx = time_stmp[(time_stmp.time > time_reversed[train_start]) & (time_stmp.time <= time_reversed[train_end])].index
            test_idx = time_stmp[(time_stmp.time > time_reversed[test_start]) & (time_stmp.time <= time_reversed[test_end])].index
            yield train_idx, test_idx




