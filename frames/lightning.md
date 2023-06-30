https://lightning.ai/docs/pytorch/stable/expertise_levels.html

- wrapper on top of PyTorch
- compact cleaner code (avoiding boilerplate code)
- multi GPU training

Lightning automates stuff like these:

- Epoch and batch iteration
- optimizer.step(), loss.backward(), optimizer.zero_grad() calls
- Calling of model.eval(), enabling/disabling grads during evaluation
- Checkpoint Saving and Loading
- multi gpu support

`import lightning.pytorch as pl`

-------------------------------------------------------------
main stuff : 
- **LightningModule**
	- model, loss, optimizer
	- train step, etc.
- **LightningDataModule**
	- dataloader, etc.
- **Trainer**
	- uses stuff in lightning module and data module

-------------------------------------------------------------
# Lightning module
a class;  inherit from `pl.LightningModule` [ Its implementation inherits from nn.Module(so everything in that is available) + adds more functionality ]
- **init** :  model objects(of usual PyTorch model classes), loss_fn[then call as `self.loss_fn` in training_step], evaluation metric (**some jars**) 
- functions: `forward`, `training_step`, `validation_step`, `test_step`, `predict_step`, `configure_optimizers` **essentially we are overriding these functions** . All pl.LightningModule, nn.Module functions can be overridden. These functions are all "hooks".
- these work along with `pl.Trainer` 
- **forward(self, x)** - similar implementation as in PyTorch models.. (inference, .. )
- **training_step(self, batch, batch_id)** - "one batch step" training (hence "step" in the name)
	> x, y = batch  
	> pred = self.forward(x)  
	> loss = self.loss_fn(pred, y)  
	> return loss  
	
	(log loss, eval metrics, etc. )   
 
	[`trainer.fit(lightning_module, train_loader/data_module)`] internally uses the training_step

- **validation_step(self, batch, batch_id)**, **test_step(self, batch, batch_id)** - similarly for "one batch step". 
	- trainer.validate(), trainer.test() internally uses validation_step, test_step
	- Data loader is not mentioned here. Pass corresponding loader in trainer. (`trainer.validate(lighting_module, val_loader/data_module`, `trainer.test(lightning_module, test_loader/data_module)`) 
	- if data_module is passed, it will automatically pick the corresponding loader

	(No opt.step, loss.backward, zero_grad in above. Automatically taken care)       
	**we specify what is to done in one "step". Other boiler plate stuff is automatically taken care**   

- **configure_optimizers**  - return optimizer object. Make optimizer object and return. Can return multiple optimizers, schedulers (as a dictionary)
	- `self.parameters` (like in nn.Module. As model blocks are class attributes) 
	- **"lightning module" can be abstracted as PyTorch model with additional functionality**

- `self.log("tag", metric_value, on_epoch=, on_step=, prog_bar=)` in any of the functions. (logged on tensorboard); `self.log_dict( {"tag":value} )`  (ie, LightningModule already has a log implementation)

- `predict_step` prediction given input. **model.predict(model, data_loader)** uses predict_step. Pre-processing, post processing can be implemented in this.


Making lightning module object - pass those need by init.


https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/core/module.py

### hooks
- `on_thing_start`, `on_thing_end` . fit, train, validation, test, predict, train_batch, validation_batch, test_batch, predict_batch, train_epoch, validation_epoch, test_epoch, predict_epoch
-  `on_thing_model_train`, `on_thing_model_train`. validation_model, test_model, predict_model
- `on_before_thing` , `on_after_thing`. zero_grad, backward, optimizer_step
- eg: `training_epoch_end(self, outputs)` - "outputs" : output at the end of each epoch, what returned by  "training_step" [that is automatically passed here] - can compute metrics, etc.. here.. eg: average over epochs, etc..   

**Any hooks can be overridden and customized.**   
https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightning-hooks

https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/core/hooks.py

-----------------------------------------------------------------
(**data - not inside lightning module**)   
- either a usual PyTorch dataset, data loader; or a lightning data module 
- usual torch_loader / lightning data_module can be passed to "trainer" (trainer.fit, trainer.validate, trainer.test, etc.) 

-----------------------------------------------------------------

# Data module

- inherit from `pl.LightningDataModule`
- "data hooks"
- `init` : data_path, batch_size, etc.
- `prepare_data` : preprocessing,..
- `setup(self, stage)`  - **the dataset class stuff** assign self.train_dataset, self.val_dataset, self.test_dataset (these will be used in the data loader hooks) [datasets be "index accessible". `getitem`, .. if needed]
- `train_dataloader`, `val_dataloader`, `test_dataloader` return corresponding loaders. Use the datasets [`self.__dataset`] (assigned in setup, make data loader and return)
-----------------------------------------------------------------

# Trainer

> trainer = pl.Trainer()   
> trainer.fit( lightning_module, train_loader, val_loader ) [both train_loader and val_loader]   
> trainer.fit( lightning_module, data_module )   
- "for e in range(epochs):  .... "<- boiler plate. This is replaced.
- like the **"general fit function"** 
- internally uses the training_step, etc. in lightning module.
- `trainer.validate(lightning_module, val_loader/data_module)` 
- `trainer.test(lightning_module, test_loader/data_module)` 
- if data_module is passed, the corresponding loader will be automatically picked
- `trainer.tune(lightning_module, test_loader)` parameter tuning
- `trainer.predict()` uses predict_step
 
 ### initializing trainer object:
- `trainer = pl.Trainer (accelerator="gpu", devices=1,  min_epochs = , max_epochs=, enable_checkpointing=, )` many such things can be chosen while making trainer object
- **callbacks**, overfit_batches, fast_dev_run(fastly check train, val, test), default_root_dir, ...
- multiple gpu, .. 


**make trainer object(many options), then call trainer.fit, trainer.validate, trainer.fit, ..**  

https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/trainer/trainer.py     
(see init for available options)

------------------------------------------------------

# project structure
- model module, data module, train
- model.py, data.py, train.py, config.py
- data.py : data module
- model.py : lightning module
- train.py 
	- import everything
	- make lightning module object, data module object, etc.
	- make trainer obj, call trainer.fit, ..
	
(separate files for custom callbacks, custom metrics, etc. )    

(**don't need class in config. just the variables will do.** import file; file.var)

-----------------------------------------------------
-----------------------------------------------------

# callbacks
- to do extra stuff ... triggered by "events" (they are always called on those events. We override them and add thing we want. )
- make a class with callbacks - inherit from `pl.callbacks.Callback` - (so that we can override the hooks)
- hooks `on_thing_start`, `on_thing_end` , ...
- passed while making trainer object   
- `trainer = pl.Trainer( callbacks = [list, of, callback, objects] )`
 


# eval metrics
- "torchmetrics" package
- eg : torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
- assign as a class attribute, call self.fn when needed

### custom metric:
- class my_metric(Metric): (`torchmetrics.Metric`) 
- "states" (values tracked during training) eg: total, correct, etc.
-  add states in init using self.add_state eg: `self.add_state("state_tag", default=, dist_reduce_fx="sum")`
- update `self.state_tag` in update eg: `update(self, preds, targets)`
- `compute` final computation with the final updated values. return the needed value.
- calling metric_object calls `metric_object.update()` and `my_metric.compute()`. use as needed(in training_step, .. ) eg: `self.metric_object(preds, targets)` (inputs for update function)

# tensorboard
- `pl.logger.TensorBoardLogger`
- there is WandbLogger also
- logger object with path, .. `logger = TensorBoardLogger("path", name="")`
- pass to trainer `trainer = pl.Trainer(logger=logger)`
- automatically logs 'logged' values to tensorboard
------------------------------------------------------
- in "lightning module" - `self.logger` to log more things. (other that `self.log,` `self.log_dict`)
- eg: `self.logger.experiment.add_image("tag", image)`...  
- the lightning module 'logger' is a general logger thing that can be used to log different things. This logger is not same as the logger passed to trainer object. That will be tensorboard logger or wandb logger, etc. That takes what is logged by the general logger in the lightning module.
- saved to current working directory. or set "default_root_dir" in Trainer.

Then usual tensorboard stuff.

(check how things are inheriting, etc.. in the source code to get an idea.. )

# profiling
- where is compute time spent
- pass while making trainer object (`trainer=pl.Trainer(profiler=profiler)`)
- inbuilt profilers ( eg: SimpleProfiler -  `Trainer(profiler="simple")` )
- other profilers using `pl.profilers.PyTorchProfiler`

------------------------------------------------------

multi GPU stuff.. later


------------------------------------------------------
------------------------------------------------------
- Lightning module attribute can be another lightning module object. (any object can be class attribute)
- `module.freeze()` freeze all parameters of model



**Model summary:**    
- `pl.utilities.model_summary.ModelSummary`
- `print( ModelSummary(lightning_module, max_depth=-1) )`
- to add child modules also to summary - add a ModelSummary callback to trainer
-  `pl.callbacks.ModelSummary` ; `trainer = Trainer(callbacks=[ModelSummary(max_depth=-1)])`
- `Trainer(enable_model_summary=False)` to turn it off   

**Checkpoint:**
- `trainer = Trainer(enable_checkpointing=True)` This is True by default.
- `trainer = Trainer(default_root_dir="some/path/")` saves checkpoint to that path every epoch end
- `model=lightning_module.load_from_checkpoint("checkpoint_path", init args)` this returns "lightning module" with weights loaded in model; `model.eval()` ; `model(x)` (**lightning module can do module.eval(), module(x),.. as it inherits from nn.Module**)
- The saved checkpoint is the usual one. So`torch.load` also works.
- `trainer.fit(model, ckpt_path="checkpoint/path")`  automatically restores model, epoch, step, LR schedulers, etc.
- checkpoint "callbacks"    
- `self.save_hyperparameters()` in lightning module - will save hyper parameters also("passed to init").  Saved to "hyper_parameters"  key of checkpoint. `loaded_checkpoint["hyper_parameters"]` to access after loading checkpoint. 




**Quick check:**   
- `Trainer(fast_dev_run=5)` run only 5 batches (with checkpointing disabled, .. ) to check for errors, .. 
- `Trainer(limit_train_batches=0.1, limit_val_batches=0.01)` use only 10% of training data and 1% of val data
- `Trainer(limit_train_batches=10, limit_val_batches=5)` use 10 batches of train and 5 batches of val
- `Trainer(num_sanity_val_steps=2)` runs 2 steps of validation in the beginning of training


All these while making Trainer object.

**early stopping**   
- early stopping callback
- `pl.callbacks.early_stopping.EarlyStopping`
- eg: `early_stop_callback=EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")`
- custom early stopping callback : make custom callback class; inherit from "EarlyStopping"; override needed functions like on_validation_end, on_train_end, .. 




------------------------------------------------------
------------------------------------------------------
**Lightning module :- model; loss; evaluation; train,val,test**     
Out of data, model, loss, optimizer :- except data, everything else in lightning module; along with training, validation, test steps. This + "data"(usual data loader or data module) is passed to Trainer.

- `__init__` -> assign models, loss, evaluation metrics, .. (then use where need in other attributes)
- `configure_optimizers`
- `training_step`
- `validation_step`
- `test_step`


**trainer.fit**      
- input : lightning module, train loader, val loader(optional)
- 

**callbacks.LearningRateMonitor**     
- logs current learning rate (comes in tensorboard)
- `trainer=pl.Trainer( callbacks=[LearningRateMonitor("epoch")] )`


**callbacks.ModelCheckpoint**      
- allows you to customize the saving routine of your checkpoints : how many checkpoints to keep, when to save, which metric to look out for, etc.
- `trainer=pl.Trainer( callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"))`
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTkwNTk2NTAwNSwxMzkzMDY1MywxODc5MT
YyOTIsLTEwMDEzNTEyMTcsNjY1MTIwMTgxLDEwNjM1NzU5OTdd
fQ==
-->