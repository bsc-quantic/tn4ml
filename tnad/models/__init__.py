from quimb import tensor as qtn
from tqdm import tqdm
import funcy
import numpy as np


def lambda_value(lambda_init=1e-3, epoch=0, decay_rate=0.01):
    return lambda_init * np.power((1 - decay_rate / 100), epoch)


class Model:

    # NOTE data already embedded
    def configure(self, loss, strategy="dmrg", optimizer="adam", **kwargs):
        self.loss = loss

        if isinstance(strategy, Strategy):
            pass
        elif strategy in ["sweeps", "local", "dmrg"]:
            strategy = Sweeps()  # TODO
        elif strategy in ["global"]:
            strategy = Global()  # TODO
        else:
            raise ValueError(f'Strategy "{strategy}" not found')
        self.strategy = strategy

        self.optimizer = optimizer

    def train(self, data, batch_size=None, epochs=1, initial_epochs=None, decay_rate=0.01, **kwargs):

        if batch_size:
            data = np.split(data, data.shape[0] // batch_size)
        else:
            data = [data]  # NOTE fixes `for batch in data`

        for epoch in (pbar := tqdm(range(epochs))):
            pbar.set_description(f"Epoch #{epoch}")
            for batch in data:
                if not isinstance(self.optimizer, str) and initial_epochs and epoch >= initial_epochs:
                    lambda_it = lambda_value(lambda_init=self.optimizer.learning_rate, epoch=epoch - initial_epochs, decay_rate=decay_rate)
                    self.optimizer.learning_rate = lambda_it
                self.fit_step(loss_fn=self.loss, loss_constants={"batch_data": batch}, **kwargs)

    def fit_step(self, loss_fn, niter=1, **kwargs):
        for sites in self.strategy.iterate_sites(self):
            # contract tensors (if needed)
            self.strategy.prehook(self, sites)

            if isinstance(loss_fn, dict):
                error = loss_fn["error"]
                data = kwargs["loss_constants"].pop("batch_data")

                loss_fn = [lambda model: error(model, sample) for sample in data]

                if "reg" in loss_fn:
                    loss_fn.append(loss_fn["reg"])

            target_site_tags = tuple(self.site_tag(site) for site in funcy.flatten(sites))
            opt = qtn.TNOptimizer(
                self,
                loss_fn=loss_fn,
                optimizer=self.optimizer,
                tags=target_site_tags,
                **kwargs,
            )

            if isinstance(self.optimizer, str):
                optself = opt.optimize(niter)
                self._tensors = optself.tensors
            else:
                x = opt.vectorizer.vector
                _, grads = opt.vectorized_value_and_grad(x)
                grads = opt.vectorizer.unpack(grads)

                tensors = self.select_tensors(target_site_tags, which="any")
                for tensor, grad in zip(tensors, grads):
                    tensor.modify(data=self.optimizer(tensor.data, grad))

            # split tensors (if needed) & renormalize (if configured)
            self.strategy.posthook(self, sites)

    def predict(self, x):
        return self @ x


from .smpo import SpacedMatrixProductOperator
