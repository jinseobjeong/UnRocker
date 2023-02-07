
class CustomLoss():
        def __init__(self, steps_per_epoch):
            super().__init__()
            self.steps_per_epoch = steps_per_epoch
            self.step = 0

        def calc_custom_loss(self, y_true, y_pred):
            y_true1 = get_y_true1(self.step)
            y_true2 = get_y_true2(self.step)

            self.step += 1
            self.step %= self.steps_per_epoch
