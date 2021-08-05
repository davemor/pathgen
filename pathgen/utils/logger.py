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
        self.current_epoch = 0
    
    def __call__(self, key, value):
        if key not in self.variables:
            self.variables[key] = LoggedVariable()
        self.variables[key].append(value)
    
    def end_epoch(self):
        self.current_epoch += 1
        for _, val in self.variables.items():
            val.end_epoch()

    def print_summary_of_latest_epoch(self):
        print('\r', f"epoch {self.current_epoch}", end = '\t')
        for key, val in self.variables.items():
            print(f" {key}: {val.epoch_values[-1]:.2f} ", end = '\t')
        print()
    
    def history(self):
        return { k:v.epoch_values for k, v in self.variables.items() }