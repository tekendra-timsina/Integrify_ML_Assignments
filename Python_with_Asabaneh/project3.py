class StatSummary:
    def __init__(self,values):
        self.values = values
        self.index = len(values)

    def mean(self):
        return round(sum(self.values)/self.index)

    def median(self):
        sorted_val = sorted(self.values)
        if self.index%2 == 0:
            return (sorted_val[round((self.index/2)-1)]+sorted_val[round(((self.index/2)+1)-1)])/2
        else:
            return sorted_val[round(((self.index+1)/2))-1]
    
    def mode(self):
        a_dict = {}
        for val in self.values:
            if val not in a_dict:
                a_dict[val] = 1
            else:
                a_dict[val] += 1
        result = sorted(a_dict.items(),key=lambda x:x[1],reverse=True)
        return f"{result[0][0]}, {result[0][1]} times"

    def min(self):
        return min(self.values)
        # return sorted(self.values)[0]

    def max(self):
        # return max(self.values)
        return sorted(self.values)[-1]

    def count(self):
        return self.index
    
    def sum(self):
        return sum(self.values)
    
    def range(self):
        return (max(self.values)-min(self.values))

    def std(self):
        mean = sum(self.values)/self.index
        sum_ = 0
        for val in self.values:
            var = (val-mean)**2
            sum_ += var
        return round((sum_/self.index)**(1/2),1)

    def var(self):
        mean = sum(self.values)/self.index
        sum_ = 0
        for val in self.values:
            var = (val-mean)**2
            sum_ += var
        return round((sum_/self.index),1)

ages = [31, 26, 34, 37, 27, 26, 32, 32, 26, 27, 27, 24, 32, 33, 27, 25, 26, 38, 37, 31, 34, 24, 33, 29, 26]
data = StatSummary(ages)
print('mean: ',data.mean())
print('median: ',data.median())
print('mode: ',data.mode())
print('min: ',data.min())
print('max: ',data.max())
print('Count: ',data.count())
print('Sum: ',data.sum())
print('Range: ',data.range())
print('std: ',data.std())
print('variance: ',data.var())