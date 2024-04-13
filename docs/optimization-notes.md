# Optimization Notes

## Optimization in PyTorch Lightning

- [LightningModule.load_model_from_checkpoint() | Docs](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.load_from_checkpoint)
- gradient checkpointing and accumulation
    - *gradient accumulation*: https://lightning.ai/docs/pytorch/stable/common/trainer.html#accumulate-grad-batches
        - at least a basic version, see docs linked in the above section if more detail is needed
- early stopping
    - see https://lightning.ai/docs/pytorch/stable/common/early_stopping.html#early-stopping
- checkpoint best models and log them to mlflow
    - See [here](https://lightning.ai/docs/pytorch/stable/common/optimization.html#learning-rate-scheduling) to customize checkpointing (i.e. when to create the model, based on what metric improvement etc.)
- mixed precision
    - this too is a Trainer parameter, in my case the interesting one would be "16-mixed"
- [ ] setup and connect mlflow to lightning modules
    - made in such a way that the mlflow tracking server is easily hot-swappable with a managed server such as one hosted by databricks; this way we can move to colab if necessary and still use mlflow
- TODO Retrieve a model from the artifacts stored in mlflow, the best model ideally, and then load it to train/make predictions

*questions for myself*:
- do we need to manually add the CLS and SEP tokens at the end of each sentence or the tokenizer does that already, if needed?

*cool stuff*:
- `fast_dev_run` and `overfit_batches` Trainer parameters to easily check if the model runs without errors


## References on Optimizing Memory Consumption

- [Comprehensive Guide to Memory Usage in pytorch | Medium](https://medium.com/deep-learning-for-protein-design/a-comprehensive-guide-to-memory-usage-in-pytorch-b9b7c78031d3)
- [Pytorch + Nvidia memory stats | Pytorch Docs](https://pytorch.org/blog/understanding-gpu-memory-1/?hss_channel=lcp-78618366)
    - polls memory usage and dumps it into a file that can be plotted via a Pytorch given tool, so that you can see all the spikes in gpu usage
- [Automatic Mixed Precision (amp) | PyTorch Docs](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
    - how to use amp in pytorch with little example
    - [What everyone should know about mixed precision training | PyTorch Docs](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
- [Summary of Mixed Precision Training vs 16bit training - model.half() vs cuda.amp.autocast() | StackOverflow](https://stackoverflow.com/questions/69994731/what-is-the-difference-between-cuda-amp-and-model-half)
- [Optimizing LLM memory with transformers library | Huggingface Docs](https://huggingface.co/docs/transformers/v4.18.0/en/performance)
    - even if we'll probably decide to stick with pyutorch stuff, this transformers library walkthrough to memory optimization is very useful to understand how to fit realtively large models into relatively small VRAMs
        - talks about gradient accumulation, checkpointing, less memory-hungry (such as Adam) optimizers and fp16 + mixed precision training
    - Great subsection that explains in detail how much memory is consumed due to model, gradients, otpimizers, etc. and much more: [Model Scalability](https://huggingface.co/docs/transformers/v4.18.0/en/performance#model-scalability)
- [Gradient Accumulation Summary | Medium](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)
    - what is not explained here is that the cost of using gradient accumulation is doing multiple forward and backward passes, as opposed to just one in the normal case. This intuitively increases comp. time because more iterations on the model are performed, even if the gradient calculations are the same.
- [Gradient Checkpointing Explained Visually](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
    - the gifs show very well that checkpointing makes it so that you do not have to restart from the beginning, but rather from the last checkpoint, to recalculate activations and hence gradients during the backward pass.
    - it also describes various strategies for checkpointing

**Da capire come abilitare gradient accumulation e checkpointing in maniera piu' granulare sul modello pre-trainato. O forse anche no, perche' se usiamo LoRA, i pesi del modello pre-trainato rimangono sempre freezati, dunque possiamo gestire meglio cosa fare sui pesi del LoRA**
