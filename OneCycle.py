import math

class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do one cycle during 
    whole run with 2 steps of equal length. During first step, increase the learning rate 
    from lower learning rate to higher learning rate. And in second step, decrease it from 
    higher to lower learning rate. This is Cyclic learning rate policy. Author suggests one 
    addition to this. - During last few hundred/thousand iterations of cycle reduce the 
    learning rate to 1/100th or 1/1000th of the lower learning rate.

    Also, Author suggests that reducing momentum when learning rate is increasing. So, we make 
    one cycle of momentum also with learning rate - Decrease momentum when learning rate is 
    increasing and increase momentum when learning rate is decreasing.

    Args:
        nb              Total number of iterations including all epochs

        max_lr          The optimum learning rate. This learning rate will be used as highest 
                        learning rate. The learning rate will fluctuate between max_lr to
                        max_lr/div and then (max_lr/div)/div.

        momentum_vals   The maximum and minimum momentum values between which momentum will
                        fluctuate during cycle.
                        Default values are (0.95, 0.85)

        prcnt           The percentage of cycle length for which we annihilate learning rate
                        way below the lower learnig rate.
                        The default value is 10

        div             The division factor used to get lower boundary of learning rate. This
                        will be used with max_lr value to decide lower learning rate boundary.
                        This value is also used to decide how much we annihilate the learning 
                        rate below lower learning rate.
                        The default value is 10.

        use_cosine      Use cosine annealation instead of linear to change learning rate and 
                        momentum. 
                        The default value is False
    """
    def __init__(self, nb, max_lr, momentum_vals=(0.95, 0.85), prcnt= 10, div=10, use_cosine=False):
        self.nb = nb
        self.div = div
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.use_cosine = use_cosine
        if self.use_cosine:
            self.prcnt = 0
        else:
            self.prcnt = prcnt
        self.iteration = 0
        self.lrs = []
        self.moms = []
        if self.use_cosine:
            self.step_len =  int(self.nb / 4)
        else:
            self.step_len =  int(self.nb * (1- prcnt/100)/2)
        
    def calc(self):
        if self.use_cosine:
            lr = self.calc_lr_cosine()
            mom = self.calc_mom_cosine()
        else:
            lr = self.calc_lr()
            mom = self.calc_mom()
        self.iteration += 1
        return (lr, mom)
        
    def calc_lr(self):
        if self.iteration ==  0:
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        elif self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        elif self.iteration > 2 * self.step_len:
            ratio = (self.iteration - 2 * self.step_len) / (self.nb - 2 * self.step_len)
            #lr = self.high_lr * ( 1 - 0.99 * ratio)/self.div
            lr = (self.high_lr / self.div) * (1- ratio * (1 - 1/self.div))
        elif self.iteration > self.step_len:
            ratio = 1- (self.iteration -self.step_len)/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        else :
            ratio = self.iteration/self.step_len
            lr = self.high_lr * (1 + ratio * (self.div - 1)) / self.div
        self.lrs.append(lr)
        return lr

    def calc_mom(self):
        if self.iteration == 0:
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration > 2 * self.step_len:
            mom = self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else :
            ratio = self.iteration/self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        self.moms.append(mom)
        return mom

    def calc_lr_cosine(self):
        if self.iteration ==  0:
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        elif self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/(self.nb - self.step_len)
            lr = (self.high_lr/self.div) + 0.5 * (self.high_lr - self.high_lr/self.div) * (1 + math.cos(math.pi * ratio))
        else :
            ratio = self.iteration/self.step_len
            lr = self.high_lr - 0.5 * (self.high_lr - self.high_lr/self.div) * (1 + math.cos(math.pi * ratio))
        self.lrs.append(lr)
        return lr

    def calc_mom_cosine(self):
        if self.iteration == 0:
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration == self.nb:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/(self.nb - self.step_len)
            mom = self.high_mom - 0.5 * (self.high_mom - self.low_mom) * (1 + math.cos(math.pi * ratio))
        else :
            ratio = self.iteration/self.step_len
            mom = self.low_mom + 0.5 * (self.high_mom - self.low_mom) * (1 + math.cos(math.pi * ratio))
        self.moms.append(mom)
        return mom
