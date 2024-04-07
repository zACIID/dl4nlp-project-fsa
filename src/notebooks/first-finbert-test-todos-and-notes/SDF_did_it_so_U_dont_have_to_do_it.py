# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="JKPGam770FXb"
# ## FinancialBERT

# %% [markdown] id="t62U8Km5DamU"
# *References*
# 1. [FinBERT Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/bert#transformers.BertForMaskedLM)
# 2. [Tokenizer Documentation](https://huggingface.co/docs/transformers/en/internal/tokenization_utils)

# %% id="YScJEQBL0Dx0"
import torch as th
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerBase, BertForMaskedLM
import transformers.tokenization_utils_base as ttu
import transformers.modeling_outputs as tm

# %% colab={"base_uri": "https://localhost:8080/", "height": 320, "referenced_widgets": ["8d2e4a7b86124b47a37c1dc7c22578a8", "1d06301c1c54481dbac98cdb4db8ddba", "a846bc9c94f64fe59a83921599b045d4", "4d86278a0e194fd2b4a37e536338add8", "f914bee65da7493e87752b94bcccc623", "c3ab78ee888c45f097eb433da656b28f", "42e56b315dd24e258df0052e97b95e3e", "d691005c06324a75854e4adbabe4b302", "15cb932b15254201a12c22a8c13c1446", "69f0061713ad498d8eac8043c790feb5", "72f6126beb46444b8b721f778476683f", "8fed760b71ea4de1864e12ba69aa4aec", "e6be1bc1dc6442429ed7da056607ce8e", "a8056ba777384de590c677c3be0c59c7", "bc143430dd3045cbacd0b2e4ba6373a2", "7b1e3441db744af4b521424edbb5df72", "49cd82e74cde4bcdb28c0a2f12b93941", "20b85d34f8d142448c12b0a85f702f84", "25a8ae02ad764a2e8b34934487530bf5", "904c6a4e89b146f2b68dd6ef92bd5b46", "3df069b22f2e4aa49a1c50410b28a85c", "091eec765a7e4a23bd88006650699c11", "9bb461945328430382a45bfd2c53379f", "65c21b862bac4a589692f734326ca172", "a48fc7c723e8471a959feae31ab13b21", "2a0cb1f4a2584716827d80de21d53a3a", "686d3769bb1047f6a1eb5683c64a67a2", "30127c5cb5be4345a84f11f980ebcb92", "5d065d9ca592430787a7b8a51b01843b", "80123a308858419c98d70fd7f1cfa51f", "abdb7c6c0c6a4266a6be49f62ecf3d53", "6e94b87085b749cf9489f4038b726fdb", "8092de04504b4ce691cd0514b53af696", "6796ec45dbef4b1fa6a742f95cff0816", "83d66189d9c6483788b6ff6db86221f5", "4742e1113eb64ba697cdd9d8b221ac0e", "b178992a80f74355b89dcdce658f6445", "df91806e48ec45139f9ad3cb8540dc05", "a30f50c8589949fa8f0a991c7df7e249", "48d4a2e052df4a2cb419bfcc2f13a8a9", "93b05f0472b24968882a52a92f557454", "8ebd7cd5cb8a4585929660cd98912423", "761f832826744bb2adee070dd7b6ec38", "cecf660cbd864a40a54e822afc2fc944", "7b64d08d3a9f48dc85a82facd45b722a", "10013986186e426aa153c8e50c9d7b70", "aa948f2da8c14f06b3ea3951048d6124", "e0ac78ead5f44db2b72e4e7445aca9bd", "7e9ab26a74b14210aa2be0cbe8ef9865", "cb808695aebb4143a8bc30f2ce3b17e8", "0c1958d9378f4a1e871c57fcd95e6f0c", "0ca94e591baa47a49d76433df472fea7", "4eba56a785ae4e56ac8394e2b9f2a121", "056b1669903a4f44a338483fe47c86df", "a2665d1adae84835bba9e30f19e5f5a0", "b6642c8e07d5489e859ed80eb204680f", "46861198d87d41c2a66b4553c5132958", "5dfe65e5ba584d43b05d6948e2376908", "8aabe26c4bf84528995c1c99957babb0", "0b57e80c3e2b4a2a8727e1689a74b75f", "9ef46fc170f94fe2af5d15543c567a15", "fb94540f82fc4f84965c06d21e532d8d", "4966196d15814c179aff3c11b5d96315", "132bebd5871b433db8e40d0e9e4520ea", "8593d385807741e1b65ea1d0e3743a15", "29e3630c297042b086d1d65c615077ac"]} id="M5S50x95BrWg" outputId="53637fe9-c953-4b62-edf8-5543abdac846"
tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained("ahmedrachid/FinancialBERT")
model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained("ahmedrachid/FinancialBERT")

# %% [markdown] id="4qbqr5w5D2tU"
# ## Tokenize sequences

# %% [markdown] id="yvJ8SjlIKNLZ"
# So here given a string or a batch of string (list[str]) the tokenizer return an object containing:
# 1. `input_ids` (plus two for [CLS] and [SEP] if not present)
# 2. `token_type_ids` needed to understand if it's a normal word or a special token
# 3. `attention_mask` needed to understand if the model should accept at certain position the token [MASK]

# %% colab={"base_uri": "https://localhost:8080/"} id="9oUSNeziCQxY" outputId="ec1688c7-6594-4e00-8292-37f7af0d6942"
fst_sentence: str = "let's touch them kids"

fst_sample: ttu.BatchEncoding = tokenizer(text=fst_sentence, return_tensors="pt") # bruv pt stands for pytorch

fst_sample

# %% colab={"base_uri": "https://localhost:8080/"} id="QO-WGK9VLeFx" outputId="92d500da-c7ab-4177-e645-c88e716fa315"
sdn_sentence: str = "hope the loss goes down"

# Had to add padding parameter to adjust the lenght of all the sentences to the longest one
sdn_sample: ttu.BatchEncoding = tokenizer(text=[fst_sentence, sdn_sentence], padding=True, return_tensors="pt")

sdn_sample

# %% [markdown] id="YImdtExGECub"
# ## Feeding the model

# %% [markdown] id="dIffsmhhPpFO"
# The model return a matrix of size $(d\times t \times |V|)$ where $d$ is the batch size $t$ is the number of token and $|V|$ is the cardinality of the vocabulary

# %% colab={"base_uri": "https://localhost:8080/"} id="0ONKJserED12" outputId="a6090754-30ee-4efe-f4c3-e02e653d9b27"
with th.no_grad():
    logits: th.Tensor = model(**fst_sample).logits  # with this i can have the raw logits before softmax

logits.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="b1CkN23l5QNZ" outputId="f27abcab-dc10-4251-a526-e7743e8e932c"
with th.no_grad():
    logits: th.Tensor = model(**sdn_sample).logits  # with this i can have the raw logits before softmax

logits.shape

# %% [markdown] id="7vdJz8dc7NZS"
# Clearly we can shove all those logits in our butt, what we do care are the embeddings $e_i \in \mathbb{R}^{H}$ $\forall i=1, \dots, T$. In order to get them we just need to set the parameter `output_hidden_states=True`, and the model will add inside the object of type `tm.MaskedLMOutput` the `hidden_states` field. As written in the documentation the `hidden_states` is of type `tuple(torch.FloatTensor)`, where the size of tuple is 13 or 25 (because of the 12 or 24 encoders in the architecture each of which output its own embeddings plus the initial embeddings). Each entry of the tuple is of size $(d \times T \times H)$ where $H$ is the embedding size (depending on which model FinBERT was built on, in case of $BERT_{LARGE}$ $H=1024$ otherwise with $BERT_{SMALL}$ $H=768$)

# %% colab={"base_uri": "https://localhost:8080/"} id="04Sj-qBPQUHE" outputId="e919c426-e144-4bce-8566-29c10fd425d0"
with th.no_grad():
    fst_res: tm.MaskedLMOutput = model(**fst_sample, output_hidden_states=True)

fst_res.hidden_states[12].shape

# %% colab={"base_uri": "https://localhost:8080/"} id="L1rLhpnK-ZlL" outputId="bca9e909-c910-4e1c-86b6-0b3ed0c03636"
with th.no_grad():
    sdn_res: tm.MaskedLMOutput = model(**sdn_sample, output_hidden_states=True)

sdn_res.hidden_states[12].shape

# %% [markdown]
# ## MIEI TEST TODOS

# %%
len(list(model.parameters()))

# %%
th.cuda.is_available()

# %%
th.version.cuda


# %%
def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        # table.add_row([name, params])
        print(f"{name} : {params}")
        total_params += params
    # print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

count_parameters(model)

# %%
device = th.device('cuda')

# transfer model
model_cuda = model.to(device)
model_cuda.gradient_checkpointing_enable()

# %%
big_sample = ["mi piacciono i bambini e lo so che anche a te piacciono molto - mi piacciono i bambini e lo so che anche a te piacciono molto"]*128
big_sample_tokenized: ttu.BatchEncoding = tokenizer(text=big_sample, padding=True, return_tensors="pt").to(device)


# %%
# with th.no_grad():
#     logits: th.Tensor = model_cuda(**big_sample_tokenized).logits  # with this i can have the raw logits before softmax
# 
# logits.shape


# %%
logits: th.Tensor = model_cuda(**big_sample_tokenized).logits  # with this i can have the raw logits before softmax


# %%
for p in model_cuda.parameters():
    print(p.dtype)

# %%

# %% [markdown]
# ## Notes on Memory Consumption
#
# - [Comprehensive Guide to Memory Usage in pytorch | Medium](https://medium.com/deep-learning-for-protein-design/a-comprehensive-guide-to-memory-usage-in-pytorch-b9b7c78031d3)
# - [Pytorch + Nvidia memory stats | Pytorch Docs](https://pytorch.org/blog/understanding-gpu-memory-1/?hss_channel=lcp-78618366)
#     - polls memory usage and dumps it into a file that can be plotted via a Pytorch given tool, so that you can see all the spikes in gpu usage  
# - [Automatic Mixed Precision (amp) | PyTorch Docs](https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/)
#     - how to use amp in pytorch with little example  
#     - [What everyone should know about mixed precision training | PyTorch Docs](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/)
# - [Summary of Mixed Precision Training vs 16bit training - model.half() vs cuda.amp.autocast() | StackOverflow](https://stackoverflow.com/questions/69994731/what-is-the-difference-between-cuda-amp-and-model-half)
# - [Optimizing LLM memory with transformers library | Huggingface Docs](https://huggingface.co/docs/transformers/v4.18.0/en/performance)
#     - even if we'll probably decide to stick with pyutorch stuff, this transformers library walkthrough to memory optimization is very useful to understand how to fit realtively large models into relatively small VRAMs
#         - talks about gradient accumulation, checkpointing, less memory-hungry (such as Adam) optimizers and fp16 + mixed precision training
#     - Great subsection that explains in detail how much memory is consumed due to model, gradients, otpimizers, etc. and much more: [Model Scalability](https://huggingface.co/docs/transformers/v4.18.0/en/performance#model-scalability)
# - [Gradient Accumulation Summary | Medium](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)
#     - what is not explained here is that the cost of using gradient accumulation is doing multiple forward and backward passes, as opposed to just one in the normal case. This intuitively increases comp. time because more iterations on the model are performed, even if the gradient calculations are the same. 
# - [Gradient Checkpointing Explained Visually](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9)
#     - the gifs show very well that checkpointing makes it so that you do not have to restart from the beginning, but rather from the last checkpoint, to recalculate activations and hence gradients during the backward pass.    
#     - it also describes various strategies for checkpointing  
#
# **Da capire come abilitare gradient accumulation e checkpointing in maniera piu' granulare sul modello pre-trainato. O forse anche no, perche' se usiamo LoRA, i pesi del modello pre-trainato rimangono sempre freezati, dunque possiamo gestire meglio cosa fare sui pesi del LoRA**

# %%
