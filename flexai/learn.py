class Learner():
    def __init__(self, model, dataloaders, optimizer, metrics, cbs=None):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.metrics = metrics
        self.cbs = cbs
        self._epoch = 1
        self._keep_fit = True
        self._runCBS('on_init')

    @property
    def epoch(self):
        return self._epoch
    
    def fit(self, epochs, train=True, valid=True, transform=None, target_transform=None):
        phases = []
        if train: phases.append('train')
        if valid: phases.append('valid')
        self._runCBS('before_fit')
        for _ in range(epochs):
            for phase in phases:
                if phase == 'train':
                    self.model.train()
                    self.optimizer.zero_grad()
                else:
                    self.model.eval()
                self._runCBS('before_epoch', phase)
                for X, y in self.dataloaders[phase]:
                    self._runCBS('before_batch', phase, [X, y])
                    output = self.model(X)
                    loss = self.metrics['loss'](output, y)
                    m_value = {}
                    for key, func in self.metrics.items():
                        m_value[key] = func(output, y).item()
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                    self._runCBS('after_batch', phase, m_value)
                    if self._stop_fit():
                        self._runCBS('after_fit')
                        return
                self._runCBS('after_epoch', phase)
            self._epoch += 1
        self._runCBS('after_fit')
    
    def stop_fit_request(self):
        self._keep_fit = False

    def _runCBS(self, name, *args):
        if self.cbs:
            for cb in self.cbs:
                if hasattr(cb, name):
                    getattr(cb, name)(self, *args)

    def _stop_fit(self):
        r_value = not self._keep_fit
        self._keep_fit = True
        return r_value