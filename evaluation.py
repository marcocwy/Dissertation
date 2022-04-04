import sklearn as sk
import math

class Evaluation:

    def __init__(self, pred, true):
        self.pred = pred
        self.true = true

    def mae(self):
        mae = sk.mean_absolute_error(self.true, self.pred)
        return mae

    def rmse(self):
        rmse = math.sqrt(sk.metrics.mean_squared_error(self.true, self.pred))
        return rmse

    def r(self):
        r2 = r2_score(self.true, self.pred)
        return r2