from statistics import mean

class LoggedVariable:
    def __init__(self):
        self.batch_values = []
        self.epoch_values = []
        
    def append(self, value):
        self.batch_values.append(value)
        
    def end_epoch(self):
        mean_batch_values = mean(self.batch_values)
        self.epoch_values.append(mean_batch_values)
        self.batch_values = []
        
class Logger:
    def __init__(self):
        self.variables = {}
    
    def __call__(self, key, value):
        if key not in self.variables:
            self.variables[key] = LoggedVariable()
        self.variables[key].append(value)
    
    def end_epoch(self, epoch):
        print(f"end epoch {epoch}", end = '')
        for key, val in self.variables.items():
            val.end_epoch()
            print(f" {key}: {val.epoch_values[epoch]:.2f}", end = '')
        print()
    
    def history(self):
        return { k:v.epoch_values for k, v in self.variables.items() }