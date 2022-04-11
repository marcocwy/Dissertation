import sklearn as sk
import math

class Evaluation:

    def __init__(self, pred):
        self.pred = pred

    def mae(self):
        n = len(self.pred)
        sae = 0   #sum absolute errors
        for pair in self.pred:
            ae = abs(pair[0] - pair[1]) # absolute errors
            
        mae = sae / n
        return mae

    def rmse(self):
        n = len(self.pred)
        sase = 0   #sum absolute squared errors
        for pair in self.pred:
            ase = (pair[0] - pair[1])**2 # absolute squared errors 
            sase += ase

        rmse = math.sqrt(sase / n)
        return rmse

    def r(self):
        n = len(self.pred)
        sxy, sx, sy, sxs, sys = 0, 0, 0, 0, 0
        for pair in self.pred:
            x = pair[0]
            y = pair[1]

            sxy += x * y
            sx  += x
            sy  += y
            sxs += x**2
            sys += y**2

        ssr = (n * sxy) - (sx * sy)   #sum squared regression
        sst = math.sqrt((n * sxs) - sx**2) * math.sqrt((n * sys) - sy**2)   #total sum of suares
        rmse = (ssr / sst)**2
        return rmse