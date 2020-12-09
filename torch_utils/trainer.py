import pathlib
import numpy as np
import torch
from torch_lr_finder import LRFinder
from torch.utils.tensorboard import SummaryWriter
import tqdm
from tqdm.notebook import tqdm
import pickle


def save_dict(path, name, _dict):
    with open(path / f"{name}.pickle", "wb") as handle:
        pickle.dump(_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TorchTrainer:
    def __init__(self, name, model, optimizer, loss_fn, scheduler, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.name = name
        self.checkpoint_path = pathlib.Path(
            kwargs.get("checkpoint_folder", f"./models/{name}_chkpts")
        )
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.train_checkpoint_interval = kwargs.get("train_checkpoint_interval", 1)
        self.max_checkpoints = kwargs.get("max_checkpoints", 25)
        self.writer = SummaryWriter(f"runs/{name}")
        self.scheduler_batch_step = kwargs.get("scheduler_batch_step", False)
        self.additional_metric_fns = kwargs.get("additional_metric_fns", {})
        self.additional_metric_fns = self.additional_metric_fns.items()
        self.pass_y = kwargs.get("pass_y", False)
        self.valid_losses = {}

    def _get_checkpoints(self, name=None):
        checkpoints = []
        checkpoint_path = (
            self.checkpoint_path
            if name is None
            else pathlib.Path(f"./models/{name}_chkpts")
        )
        for cp in self.checkpoint_path.glob("checkpoint_*"):
            checkpoint_name = str(cp).split("/")[-1]
            checkpoint_epoch = int(checkpoint_name.split("_")[-1])
            checkpoints.append((cp, checkpoint_epoch))
        checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
        # self.valid_losses = pd.read_pickle(self.checkpoint_path/'valid_losses.pickle')
        return checkpoints

    def _clean_outdated_checkpoints(self):
        checkpoints = self._get_checkpoints()
        if len(checkpoints) > self.max_checkpoints:
            checkpoints = sorted(checkpoints, key=lambda x: x[1], reverse=True)
            for delete_cp in checkpoints[self.max_checkpoints :]:
                delete_cp[0].unlink()
                print(f"removed checkpoint of epoch - {delete_cp[1]}")

    def _save_checkpoint(self, epoch, valid_loss=None):
        self._clean_outdated_checkpoints()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": [o.state_dict() for o in self.optimizer]
            if type(self.optimizer) is list
            else self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            checkpoint.update(
                {
                    "scheduler_state_dict": [o.state_dict() for o in self.scheduler]
                    if type(self.scheduler) is list
                    else self.scheduler.state_dict()
                }
            )
        if valid_loss:
            checkpoint.update({"loss": valid_loss})
        torch.save(checkpoint, self.checkpoint_path / f"checkpoint_{epoch}")
        save_dict(self.checkpoint_path, "valid_losses", self.valid_losses)
        print(f"saved checkpoint for epoch {epoch}")
        self._clean_outdated_checkpoints()

    def _load_checkpoint(self, epoch=None, only_model=False, name=None):
        if name is None:
            checkpoints = self._get_checkpoints()
        else:
            checkpoints = self._get_checkpoints(name)
        if len(checkpoints) > 0:
            if not epoch:
                checkpoint_config = checkpoints[0]
            else:
                checkpoint_config = list(filter(lambda x: x[1] == epoch, checkpoints))[
                    0
                ]
            checkpoint = torch.load(checkpoint_config[0])
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if not only_model:
                if type(self.optimizer) is list:
                    for i in range(len(self.optimizer)):
                        self.optimizer[i].load_state_dict(
                            checkpoint["optimizer_state_dict"][i]
                        )
                else:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scheduler is not None:
                    if type(self.scheduler) is list:
                        for i in range(len(self.scheduler)):
                            self.scheduler[i].load_state_dict(
                                checkpoint["scheduler_state_dict"][i]
                            )
                    else:
                        self.scheduler.load_state_dict(
                            checkpoint["scheduler_state_dict"]
                        )
            print(f'loaded checkpoint for epoch - {checkpoint["epoch"]}')
            return checkpoint["epoch"]
        return None

    def _load_best_checkpoint(self):
        if self.valid_losses:
            best_epoch = sorted(self.valid_losses.items(), key=lambda x: x[1])[0][0]
            loaded_epoch = self._load_checkpoint(epoch=best_epoch, only_model=True)

    def _step_optim(self):
        if type(self.optimizer) is list:
            for i in range(len(self.optimizer)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _step_scheduler(self, valid_loss=None):
        if type(self.scheduler) is list:
            for i in range(len(self.scheduler)):
                if self.scheduler[i].__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler[i].step(valid_loss)
                else:
                    self.scheduler[i].step()
        else:
            if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

    def _loss_batch(self, xb, yb, optimize, pass_y, additional_metrics=None):
        if type(xb) is list:
            xb = [xbi.to(self.device) for xbi in xb]
        else:
            xb = xb.to(self.device)
        yb = yb.to(self.device)
        if pass_y:
            y_pred = self.model(xb, yb)
        else:
            y_pred = self.model(xb)
        loss = self.loss_fn(y_pred, yb)
        if additional_metrics is not None:
            additional_metrics = [fn(y_pred, yb) for name, fn in additional_metrics]
        if optimize:
            loss.backward()
            self._step_optim()
        loss_value = loss.item()
        del xb
        del yb
        del y_pred
        del loss
        if additional_metrics is not None:
            return loss_value, additional_metrics
        return loss_value

    def evaluate(self, dataloader):
        self.model.eval()
        eval_bar = tqdm(dataloader, leave=False)
        with torch.no_grad():
            loss_values = [
                self._loss_batch(xb, yb, False, False, self.additional_metric_fns)
                for xb, yb in eval_bar
            ]
            if len(loss_values[0]) > 1:
                loss_value = np.mean([lv[0] for lv in loss_values])
                additional_metrics = np.mean([lv[1] for lv in loss_values], axis=0)
                additional_metrics_result = {
                    name: result
                    for (name, fn), result in zip(
                        self.additional_metric_fns, additional_metrics
                    )
                }
                return loss_value, additional_metrics_result
            # eval_bar.set_description("evaluation loss %.2f" % loss_value)
            else:
                loss_value = np.mean(loss_values)
                return loss_value, None

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for xb, yb in tqdm(dataloader):
                if type(xb) is list:
                    xb = [xbi.to(self.device) for xbi in xb]
                else:
                    xb = xb.to(self.device)
                yb = yb.to(self.device)
                y_pred = self.model(xb)
                predictions.append(y_pred.cpu().numpy())
        return np.concatenate(predictions)

    # pass single batch input, without batch axis
    def predict_one(self, x):
        self.model.eval()
        with torch.no_grad():
            if type(x) is list:
                x = [xi.to(self.device).unsqueeze(0) for xi in x]
            else:
                x = x.to(self.device).unsqueeze(0)
            y_pred = self.model(x)
            if self.device == "cuda":
                y_pred = y_pred.cpu()
            y_pred = y_pred.numpy()
            return y_pred

    def lr_find(self, dl, optimizer=None, start_lr=1e-7, end_lr=1e-2, num_iter=200):
        if optimizer is None:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-6, momentum=0.9)
        lr_finder = LRFinder(self.model, optimizer, self.loss_fn, device=self.device)
        lr_finder.range_test(dl, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter)
        lr_finder.plot()

    def train(
        self,
        epochs,
        train_dataloader,
        valid_dataloader=None,
        resume=True,
        resume_only_model=False,
    ):
        start_epoch = 0
        if resume:
            loaded_epoch = self._load_checkpoint(only_model=resume_only_model)
            if loaded_epoch:
                start_epoch = loaded_epoch
        for i in tqdm(range(start_epoch, start_epoch + epochs), leave=True):
            self.model.train()
            training_losses = []
            running_loss = 0
            training_bar = tqdm(train_dataloader, leave=False)
            for it, (xb, yb) in enumerate(training_bar):
                loss = self._loss_batch(xb, yb, True, self.pass_y)
                running_loss += loss
                training_bar.set_description("loss %.4f" % loss)
                if it % 100 == 99:
                    self.writer.add_scalar(
                        "training loss",
                        running_loss / 100,
                        i * len(train_dataloader) + it,
                    )
                    training_losses.append(running_loss / 100)
                    running_loss = 0
                if self.scheduler is not None and self.scheduler_batch_step:
                    self._step_scheduler()
            print(f"Training loss at epoch {i + 1} - {np.mean(training_losses)}")
            if valid_dataloader is not None:
                valid_loss, additional_metrics = self.evaluate(valid_dataloader)
                self.writer.add_scalar("validation loss", valid_loss, i)
                if additional_metrics is not None:
                    print(additional_metrics)
                print(f"Valid loss at epoch {i + 1} - {valid_loss}")
                self.valid_losses[i + 1] = valid_loss
            if self.scheduler is not None and not self.scheduler_batch_step:
                self._step_scheduler(valid_loss)
            if (i + 1) % self.train_checkpoint_interval == 0:
                self._save_checkpoint(i + 1)
