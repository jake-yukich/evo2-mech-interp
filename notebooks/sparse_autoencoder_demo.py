# %%
from typing import List, Optional, Callable

import torch
import numpy as np

import os

from Bio import SeqIO

import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

from tqdm.notebook import tqdm

from evo2 import Evo2


# Ensure Triton can locate ptxas (incl. Blackwell) to avoid NoneType path errors in Triton compiler
try:
    _ptxas_path = "/opt/conda/envs/evo2/bin/ptxas"
    if os.path.isfile(_ptxas_path):
        os.environ.setdefault("TRITON_PTXAS_PATH", _ptxas_path)
        # Blackwell-capable Triton expects a different env var name
        os.environ.setdefault("TRITON_PTXAS-BLACKWELL_PATH", _ptxas_path)
        # Also ensure PATH contains the bin dir
        _bin_dir = "/opt/conda/envs/evo2/bin"
        if _bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{_bin_dir}:" + os.environ.get("PATH", "")
except Exception:
    pass


# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_grad_enabled(False)

# %% BOLIERPLATE
class ModelScope:
    """Class for adding, using, and removing PyTorch hooks with a model."""

    def __init__(self, model):
        self.model = model
        self.hooks = {}
        self.activations_cache = {}
        self.override_store = {}
        self._build_module_dict()

    """Module listing."""
    def _build_module_dict(self):
        """Walks the model's module tree and builds a name: module map."""
        self._module_dict = {}

        def recurse(module, prefix=''):
            """Recursive tree walk to build self._module_dict."""
            for name, child in module.named_children():
                self._module_dict[prefix+name] = child
                recurse(child, prefix=prefix+name+'-')

        recurse(self.model)  # build the tree

    def list_modules(self):
        """Lists all modules in the module dictionary."""
        return self._module_dict.keys()
    
    """Generic hook registration"""
    def add_hook(self, hook_fn, module_str, hook_name):
        """Add a hook_fn to the module given by module_str."""
        module = self._module_dict[module_str]
        hook_handle = module.register_forward_hook(hook_fn)
        self.hooks[hook_name] = hook_handle
    
    """Activations caching"""
    def _build_caching_hook(self, module_str):
        self.activations_cache[module_str] = []
        def hook_fn(model, input, output):
            self.activations_cache[module_str].append(output)

        return hook_fn

    def add_caching_hook(self, module_str):
        """Adds an activations caching hook at the location in module_str."""
        hook_fn = self._build_caching_hook(module_str)
        self.add_hook(hook_fn, module_str, 'cache-'+module_str)

    def clear_cache(self, module_str):
        """Clears the activations cache corresponding to module_str."""
        if module_str not in self.activations_cache.keys():
            raise KeyError(f'No activations cache for {module_str}.')
        
        else:
            self.activations_cache[module_str] = []

    def clear_all_caches(self):
        """Clear all activation caches."""
        for module_str in self.activations_cache.keys():
            self.clear_cache(module_str)

    def remove_cache(self, module_str):
        """Remove the cache for module_str."""
        del self.activations_cache[module_str]

    def remove_all_caches(self):
        """Remove all caches."""
        caches = list(self.activations_cache.keys())
        for cache_str in caches:
            self.remove_cache(cache_str)

    """Activation override"""
    def _build_override_hook(self, module_str):
        self.override_store[module_str] = None  # won't override when returned
        def hook_fn(model, input, output):
            return self.override_store[module_str]
        
        return hook_fn
    
    def add_override_hook(self, module_str):
        """Adds hook to overrides output of module_str using override_store"""
        hook_fn = self._build_override_hook(module_str)
        self.add_hook(hook_fn, module_str, 'override-'+module_str)

    def override(self, module_str, override_tensor):
        """Sets the override tensor for module_str."""
        self.override_store[module_str] = override_tensor

    def clear_override(self, module_str):
        """Clear override hook so it won't affect forward pass."""
        self.override_store[module_str] = None

    def clear_all_overrides(self):
        """Clear all override hooks."""
        overrides = list(self.override_store.keys())
        for override in overrides:
            self.clear_override(override)

    """Hook clearup"""
    def remove_hook(self, hook_name):
        """Remove a hook with name hook_name from the model."""
        self.hooks[hook_name].remove()
        del self.hooks[hook_name]

    def remove_all_hooks(self):
        """Remove all hooks from the model."""
        hooks = list(self.hooks.keys())
        for hook_name in hooks:
            self.remove_hook(hook_name)


INTERVENTION_INTERFACE = Callable[[torch.Tensor], torch.Tensor]


class ObservableEvo2:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.evo_model: NucleotideModel = Evo2(model_name)
        self.scope = ModelScope(self.evo_model.model)
        self.tokenizer = self.evo_model.tokenizer
        self.model = self.evo_model.model
        self.d_hidden = 4096

    @property
    def device(self):
        return next(self.evo_model.model.parameters()).device
        
    @property
    def dtype(self):
        return self.evo_model.dtype

    def list_modules(self):
        return self.scope.list_modules()

    def forward(
        self, 
        toks: torch.Tensor, 
        cache_activations_at: Optional[List[str]] = None, 
        interventions: dict[str, INTERVENTION_INTERFACE] = None,
    ):
        if not interventions:
            interventions = {}

        if not cache_activations_at:
            cache_activations_at = []

        output_cache = {}

        layers = list(set(list(interventions.keys()) + cache_activations_at))

        if layers:
            for layer in layers:
                def _intervene(model, input, output):
                    acts = output[0] if isinstance(output, tuple) else output

                    if layer in interventions:
                        acts = interventions[layer](acts)
                    '''
                    if layer in cache_activations_at and output_cache.get(layer, None) is None:
                        output_cache[layer] = [acts]
                    elif layer in cache_activations_at:
                        output_cache[layer].append(acts)
                    '''
                    if layer in cache_activations_at:
                        output_cache[layer] = acts.detach()
                    '''
                    if len(output) == 2:
                        return (acts, output[1])
                    else:
                        return acts
                    '''
                    return (acts, output[1]) if isinstance(output, tuple) else acts
                
                self.scope.add_hook(_intervene, layer, f'intervene-{layer}')

        # Run forwards pass
        try:
            model_outputs = self.model(toks)
            #cache = {key: output[0][0] for key, output in self.scope.activations_cache.items()}
            cached_activations = {layer: act.clone() for layer, act in output_cache.items()}
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()
                                                   
        return model_outputs[0], cached_activations #{layer: act.clone().detach() for layer, act in output_cache.items()}

    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.,
        batched: bool = True,
        cached_generation: bool = False,
        verbose: int = 0,
        cache_activations_at: Optional[List[str]] = None, 
        interventions: dict[str, INTERVENTION_INTERFACE] = None,
    ):
        #ACTIVATION_SCALING_CONSTANT = 2.742088556289673
        if not interventions:
            interventions = {}

        if not cache_activations_at:
            cache_activations_at = []

        output_cache = {}

        layers = list(set(list(interventions.keys()) + cache_activations_at))

        if layers:
            for layer in layers:
                def _intervene(model, input, output):
                    acts = output[0]

                    if layer in interventions:
                        acts = interventions[layer](acts) # * ACTIVATION_SCALING_CONSTANT) / ACTIVATION_SCALING_CONSTANT

                    if layer in cache_activations_at and output_cache.get(layer, None) is None:
                        output_cache[layer] = [acts] # * ACTIVATION_SCALING_CONSTANT]
                    elif layer in cache_activations_at:
                        output_cache[layer].append(acts)

                    if len(output) == 2:
                        return (acts, output[1])
                    else: 
                        return acts
                    # return (acts, output[1])
                
                self.scope.add_hook(_intervene, layer, f'intervene-{layer}')

        # Run forwards pass
        try:
            output = self.evo_model.generate(
                prompt_seqs,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
            )
        finally:
            self.scope.remove_all_hooks()
            self.scope.clear_all_caches()

        acts_cache = {layer: torch.cat(acts, dim=1).clone().detach() for layer, acts in output_cache.items()}
                       
        return ''.join(output[0]), acts_cache

class BatchTopKTiedSAE(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_hidden,
        k,
        device,
        dtype,
        tiebreaker_epsilon: float = 1e-6
        ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        
        W_mat = torch.randn((d_in, d_hidden))
        W_mat = 0.1 * W_mat / torch.linalg.norm(W_mat, dim=0, ord=2, keepdim=True)
        self.W = torch.nn.Parameter(W_mat)
        self.b_enc = torch.nn.Parameter(torch.zeros(self.d_hidden))
        self.b_dec = torch.nn.Parameter(torch.zeros(self.d_in))
        self.device = device
        self.dtype = dtype
        self.tiebreaker_epsilon = tiebreaker_epsilon
        self.tiebreaker = torch.linspace(0, tiebreaker_epsilon, d_hidden)
        self.to(self.device, self.dtype)
        
    def encoder_pre(self, x):
        return x @ self.W + self.b_enc

    def encode(self, x, tiebreak=False):
        f = torch.nn.functional.relu(self.encoder_pre(x))
        return self._batch_topk(f, self.k, tiebreak=tiebreak)
    
    def _batch_topk(self, f, k, tiebreak=False):
        from math import prod

        if tiebreak:  # break ties in feature order for determinism
            f += self.tiebreaker.broadcast_to(f)
        *input_shape, _ = f.shape  # handle higher-dim tensors (e.g. from sequence input)
        numel = k * prod(input_shape)
        f_topk = torch.topk(f.flatten(), numel, dim=-1)
        f_topk = torch.zeros_like(f.flatten()).scatter(-1, f_topk.indices, f_topk.values).reshape(f.shape)
        return f_topk

    def decode(self, f):
        return f @ self.W.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

def load_topk_sae(
    sae_path: str,
    d_hidden: int,
    device: str,
    dtype: torch.dtype,
    expansion_factor: int = 16,
):
    sae_dict = torch.load(sae_path, weights_only=True, map_location="cpu")

    new_dict = {}
    for key, item in sae_dict.items():
        new_dict[key.replace("_orig_mod.", "").replace("module.", "")] = item

    sae_dict = new_dict

    cached_sae = BatchTopKTiedSAE(
        d_hidden,
        d_hidden * expansion_factor,
        64, # this is a topk64 sae
        device,
        dtype,
    )
    cached_sae.load_state_dict(sae_dict)

    return cached_sae

# %%
file_path = hf_hub_download(
    repo_id=f"Goodfire/Evo-2-Layer-26-Mixed",
    filename=f"sae-layer26-mixed-expansion_8-k_64.pt",
    repo_type="model"
)
file_path
# %%
# Clear GPU cache before loading model
print("Clearing GPU cache before model loading...")
torch.cuda.empty_cache()
print(f"GPU memory before model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

model = ObservableEvo2(model_name="evo2_7b_262k")
print(f"GPU memory after model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
topk_sae = load_topk_sae(
    file_path,
    d_hidden=model.d_hidden,
    device=model.device,
    dtype=torch.bfloat16,
    expansion_factor=8
)

# Let's check what modules are available
print("Available modules in the model:")
available_modules = model.list_modules()
block_modules = [module for module in available_modules if 'blocks-' in module and module.count('-') == 1]
print("Block modules:")
for module in block_modules:
    print(f"  {module}")
print(f"Total blocks: {len(block_modules)}")

# The SAE was trained on layer 26, so let's use the 26th block (0-indexed)
if len(block_modules) > 26:
    SAE_LAYER_NAME = block_modules[26]
    print(f"Using layer: {SAE_LAYER_NAME}")
else:
    print(f"Warning: Only {len(block_modules)} blocks available, but trying to access block 26")
    SAE_LAYER_NAME = block_modules[-1] if block_modules else 'blocks-0'
    print(f"Using last available layer: {SAE_LAYER_NAME}")

# Let's also check the SAE
print(f"\nSAE info:")
print(f"  SAE d_in: {topk_sae.d_in}")
print(f"  SAE d_hidden: {topk_sae.d_hidden}")
print(f"  SAE k: {topk_sae.k}")
print(f"  SAE device: {topk_sae.device}")
print(f"  SAE dtype: {topk_sae.dtype}")
print(f"  Model d_hidden: {model.d_hidden}")
print(f"  Model device: {model.device}")
# print(f"  Model dtype: {model.dtype}")

# Check if SAE weights look reasonable
print(f"\nSAE weight stats:")
print(f"  W shape: {topk_sae.W.shape}")
print(f"  W stats: min={topk_sae.W.min():.4f}, max={topk_sae.W.max():.4f}, mean={topk_sae.W.mean():.4f}")
print(f"  b_enc stats: min={topk_sae.b_enc.min():.4f}, max={topk_sae.b_enc.max():.4f}, mean={topk_sae.b_enc.mean():.4f}")
print(f"  b_dec stats: min={topk_sae.b_dec.min():.4f}, max={topk_sae.b_dec.max():.4f}, mean={topk_sae.b_dec.mean():.4f}")

# Test SAE with random data
print(f"\nTesting SAE with random data:")
test_input = torch.randn(10, topk_sae.d_in, device=topk_sae.device, dtype=topk_sae.dtype)
print(f"  Test input shape: {test_input.shape}")
print(f"  Test input stats: min={test_input.min():.4f}, max={test_input.max():.4f}, mean={test_input.mean():.4f}")
test_output = topk_sae.encode(test_input)
print(f"  Test output shape: {test_output.shape}")
print(f"  Test output stats: min={test_output.min():.4f}, max={test_output.max():.4f}, mean={test_output.mean():.4f}")
print(f"  Test output non-zero count: {(test_output != 0).sum()}")

# Clear GPU cache
print(f"\nClearing GPU cache...")
print(f"  GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
torch.cuda.empty_cache()
print(f"  GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
# %%
def get_feature_ts(sae, seq): # Faster, but might crash
    toks = model.tokenizer.tokenize(seq)
    print(f"Tokenized sequence length: {len(toks)}")
    print(f"First 10 tokens: {toks[:10]}")
    print(f"Last 10 tokens: {toks[-10:]}")
    
    # Convert numpy uint8 to regular integers
    toks = [int(token) for token in toks]
    print(f"Converted tokens (first 10): {toks[:10]}")
    
    toks = torch.tensor(toks, dtype=torch.long).unsqueeze(0).to(model.device)
    print(f"Tensor shape after unsqueeze: {toks.shape}")
    
    # Check model state
    print(f"Model training mode: {model.model.training}")
    print(f"Grad enabled: {torch.is_grad_enabled()}")
    
    # Ensure model is in eval mode
    model.model.eval()
    print(f"Model training mode after eval(): {model.model.training}")
    
    logits, acts = model.forward(toks, cache_activations_at=[SAE_LAYER_NAME])
    print(f"Available activation keys: {list(acts.keys())}")
    
    activation_tensor = acts[SAE_LAYER_NAME]
    print(f"Raw activation tensor shape: {activation_tensor.shape}")
    print(f"Raw activation tensor stats: min={activation_tensor.min():.4f}, max={activation_tensor.max():.4f}, mean={activation_tensor.mean():.4f}")
    print(f"Raw activation tensor has NaN: {torch.isnan(activation_tensor).any()}")
    print(f"Raw activation tensor has Inf: {torch.isinf(activation_tensor).any()}")
    
    # Try both indexing approaches
    print(f"Trying acts[SAE_LAYER_NAME][0]...")
    try:
        act_slice = acts[SAE_LAYER_NAME][0]
        print(f"acts[SAE_LAYER_NAME][0] shape: {act_slice.shape}")
        print(f"acts[SAE_LAYER_NAME][0] stats: min={act_slice.min():.4f}, max={act_slice.max():.4f}, mean={act_slice.mean():.4f}")
        feats = sae.encode(act_slice)
    except Exception as e:
        print(f"Error with [0] indexing: {e}")
        print(f"Trying acts[SAE_LAYER_NAME] directly...")
        feats = sae.encode(activation_tensor)
    
    print(f"SAE output shape: {feats.shape}")
    print(f"SAE output stats: min={feats.min():.4f}, max={feats.max():.4f}, mean={feats.mean():.4f}")
    print(f"SAE output has NaN: {torch.isnan(feats).any()}")
    print(f"SAE output non-zero count: {(feats != 0).sum()}")
    
    return feats.cpu().detach().float().numpy()

def get_feature_ts_via_generate(sae, seq): # Slower, but won't crash
    logits, acts = model.generate([seq], n_tokens=1, cached_generation=True, cache_activations_at=[SAE_LAYER_NAME])
    feats = sae.encode(acts[SAE_LAYER_NAME][0])
    return feats.cpu().detach().float().numpy()
# %%
# Let's get features for 1 kb of human genome sequence (randomly selected from chr17 GRCh38.p14)
example_seq = 'TCTGAAAGGACAGTTTTATTGTAGGTACACATGGCTGCCATTTCAAATGTAACTCACAGCTTGTCCATCAGTCCTTGGAGGTCTTTCTATGAAAGGAGCTTGGTGGCGTCCAAACACCACCCAATGTCCACTTAGAAGTAAGCACCGTGTCTGCCCTGAGCTGACTCCTTTTCCAAGGAAGGGGTTGGATCGCTGAGTGTTTTTCCAGGTGTCTACTTGTTGTTAATTAATAGCAATGACAAAGCAGAAGGTTCATGCGTAGCTCGGCTTTCTGGTATTTGCTGCCCGTTGACCAATGGAAGATAAACCTTTGCCTCAGGTGGCACCACTAGCTGGTTAAGAGGCACTTTGTCCTTTCACCCAGGAGCAAACGCACATCACCTGTGTCCTCATCTGATGGCCCTGGTGTGGGGCACAGTCGTGTTGGCAGGGAGGGAGGTGGGGTTGGTCCCCTTTGTGGGTTTGTTGCGAGGCCGTGTTCCAGCTGTTTCCACAGGGAGCGATTTTCAGCTCCACAGGACACTGCTCCCCAGTTCCTCCTGAGAACAAAAGGGGGCGCTGGGGAGAGGCCACCGTTCTGAGGGCTCACTGTATGTGTTCCAGAATCTCCCCTGCAGACCCCCACTGAGGACGGATCTGAGGAACCGGGCTCTGAAACCTCTGATGCTAAGAGCACTCCAACAGCGGAAGGTGGGCCCCCCTTCAGACGCCCCCTCCATGCCTCCAGCCTGTGCTTAGCCGTGCTTTGAGCCTCCCTCCTGGCTGCATCTGCTGCTCCCCCTGGCTGAGAGATGTGCTCACTCCTTCGGTGCTTTGCAGGACAGCGTGGTGGGAGCTGAGCCTTGCGTCGATGCCTTGCTTGCTGGTGCTGAGTGTGGGCACCTTCATCCCGTGTGTGCTCTGGAGGCAGCCACCCTTGGACAGTCCCGCGCACAGCTCCACAAAGCCCCGCTCCATACGATTGTCCTCCCACACCCCCTTCAAAAGCCCCCTCCTCTCT'

# Test tokenization first
print(f"Example sequence length: {len(example_seq)}")
test_tokens = model.tokenizer.tokenize(example_seq)
print(f"Tokenized length: {len(test_tokens)}")
print(f"First 20 tokens: {test_tokens[:20]}")
print(f"Last 20 tokens: {test_tokens[-20:]}")
print(f"Unique tokens: {len(set(test_tokens))}")
print(f"Token range: {min(test_tokens)} to {max(test_tokens)}")

# Convert to regular integers
test_tokens_int = [int(token) for token in test_tokens]
print(f"Converted tokens (first 20): {test_tokens_int[:20]}")
print(f"Converted token range: {min(test_tokens_int)} to {max(test_tokens_int)}")
# Test with a simple sequence first
simple_seq = "ATCG" * 10  # 40 characters
print(f"\nTesting with simple sequence: {simple_seq}")
print(f"Simple sequence length: {len(simple_seq)}")
simple_tokens = model.tokenizer.tokenize(simple_seq)
print(f"Simple tokens: {simple_tokens}")
simple_tokens_int = [int(token) for token in simple_tokens]
print(f"Simple tokens converted: {simple_tokens_int}")

simple_feature_ts = get_feature_ts(topk_sae, simple_seq)
print(f"Simple feature_ts shape: {simple_feature_ts.shape}")
print(f"Simple feature_ts has NaN: {np.isnan(simple_feature_ts).any()}")
print(f"Simple feature_ts non-zero count: {np.count_nonzero(simple_feature_ts)}")

# Test model forward pass directly
print(f"\nTesting model forward pass directly...")
test_seq = "ATCGATCGATCG"  # 12 characters
test_tokens = model.tokenizer.tokenize(test_seq)
test_tokens_int = [int(token) for token in test_tokens]
test_tensor = torch.tensor(test_tokens_int, dtype=torch.long).unsqueeze(0).to(model.device)

print(f"Test tensor shape: {test_tensor.shape}")
print(f"Test tensor values: {test_tensor}")

# Check model parameters for NaN
print(f"\nChecking model parameters for NaN...")
has_nan_params = False
for name, param in model.model.named_parameters():
    if torch.isnan(param).any():
        print(f"  {name} has NaN values!")
        has_nan_params = True
if not has_nan_params:
    print("  No NaN values found in model parameters")

# Check model buffers for NaN
print(f"\nChecking model buffers for NaN...")
has_nan_buffers = False
for name, buffer in model.model.named_buffers():
    if torch.isnan(buffer).any():
        print(f"  {name} has NaN values!")
        has_nan_buffers = True
if not has_nan_buffers:
    print("  No NaN values found in model buffers")

# Ensure model is in eval mode
model.model.eval()

# Test forward pass
with torch.no_grad():
    logits, acts = model.forward(test_tensor, cache_activations_at=[SAE_LAYER_NAME])
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    print(f"Logits has NaN: {torch.isnan(logits).any()}")
    
    if SAE_LAYER_NAME in acts:
        raw_acts = acts[SAE_LAYER_NAME]
        print(f"Raw activations shape: {raw_acts.shape}")
        print(f"Raw activations stats: min={raw_acts.min():.4f}, max={raw_acts.max():.4f}, mean={raw_acts.mean():.4f}")
        print(f"Raw activations has NaN: {torch.isnan(raw_acts).any()}")
    else:
        print(f"Layer {SAE_LAYER_NAME} not found in activations")

# Try using the model's generate method instead
print(f"\nTesting model generate method...")
try:
    # Test with a simple sequence
    simple_gen_output = model.evo_model.generate(["ATCG"], n_tokens=10, temperature=1.0, top_k=4, top_p=1.0, batched=True, verbose=1)
    print(f"Generate output: {simple_gen_output}")
    print("Generate method works!")
except Exception as e:
    print(f"Generate method failed: {e}")

# The model appears to be corrupted. Let's try reloading it
print(f"\nModel appears corrupted - trying to reload...")
print("Clearing GPU cache...")
torch.cuda.empty_cache()

# Reload the model
print("Reloading model...")
model = ObservableEvo2(model_name="evo2_7b_262k")
print(f"Model reloaded. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Check model dtype and device
print(f"Model device: {model.device}")
print(f"Model dtype: {next(model.model.parameters()).dtype}")
print(f"Model training mode: {model.model.training}")

# Force model to eval mode and check a few parameters
model.model.eval()
print(f"Model training mode after eval: {model.model.training}")

# Check first few parameters
for name, param in list(model.model.named_parameters())[:3]:
    print(f"  {name}: shape={param.shape}, dtype={param.dtype}, device={param.device}, stats=min={param.min():.4f}, max={param.max():.4f}")

# Test the reloaded model
print(f"\nTesting reloaded model...")
test_seq = "ATCG"
test_tokens = model.tokenizer.tokenize(test_seq)
test_tokens_int = [int(token) for token in test_tokens]
test_tensor = torch.tensor(test_tokens_int, dtype=torch.long).unsqueeze(0).to(model.device)

model.model.eval()
with torch.no_grad():
    logits, acts = model.forward(test_tensor, cache_activations_at=[SAE_LAYER_NAME])
    print(f"Reloaded model logits stats: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
    print(f"Reloaded model logits has NaN: {torch.isnan(logits).any()}")

# Try using the model directly without the ObservableEvo2 wrapper
print(f"\nTesting model directly without wrapper...")
from evo2 import Evo2

# Create a fresh Evo2 model
direct_model = Evo2("evo2_7b_262k")
print(f"Direct model created. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Test direct model
test_seq = "ATCG"
test_tokens = direct_model.tokenizer.tokenize(test_seq)
test_tokens_int = [int(token) for token in test_tokens]

# Get device from model parameters
model_device = next(direct_model.model.parameters()).device
test_tensor = torch.tensor(test_tokens_int, dtype=torch.long).unsqueeze(0).to(model_device)

print(f"Direct model device: {model_device}")
print(f"Direct model dtype: {next(direct_model.model.parameters()).dtype}")
print(f"Direct model training mode: {direct_model.model.training}")

# Set to eval mode
direct_model.model.eval()

# Test forward pass
with torch.no_grad():
    try:
        # Try the model's forward method directly
        output = direct_model.model(test_tensor)
        print(f"Direct model output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Direct model output is tuple with {len(output)} elements")
            for i, elem in enumerate(output):
                if hasattr(elem, 'shape'):
                    print(f"  Element {i}: shape={elem.shape}, stats=min={elem.min():.4f}, max={elem.max():.4f}, mean={elem.mean():.4f}")
                    print(f"  Element {i} has NaN: {torch.isnan(elem).any()}")
        else:
            print(f"Direct model output shape: {output.shape}")
            print(f"Direct model output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
            print(f"Direct model output has NaN: {torch.isnan(output).any()}")
    except Exception as e:
        print(f"Direct model forward failed: {e}")

# Try using the generate method with the direct model
print(f"\nTesting direct model generate method...")
try:
    gen_output = direct_model.generate(["ATCG"], n_tokens=5, temperature=1.0, top_k=4, top_p=1.0, batched=True, verbose=1)
    print(f"Direct model generate output: {gen_output}")
    if hasattr(gen_output, 'logits') and gen_output.logits is not None:
        print(f"Direct model generate logits stats: min={gen_output.logits.min():.4f}, max={gen_output.logits.max():.4f}, mean={gen_output.logits.mean():.4f}")
        print(f"Direct model generate logits has NaN: {torch.isnan(gen_output.logits).any()}")
except Exception as e:
    print(f"Direct model generate failed: {e}")

# Now try the full sequence
print(f"\nTesting with full sequence...")
feature_ts = get_feature_ts(topk_sae, example_seq)
feature_ts.shape
# %%
# Check if there are any non-NaN, non-zero values in feature_ts
# feature_ts is assumed to be a numpy array
has_non_nan_non_zero = np.any((~np.isnan(feature_ts)) & (feature_ts != 0))
print("Has non-NaN, non-zero values in feature_ts:", has_non_nan_non_zero)

# %%
# Next, we plot a few of the features
selected_features = [15680, 28339, 1050, 25666]
fig, axes = plt.subplots(len(selected_features), 1, figsize = (30, 1*len(selected_features)), sharex = True)
for ind, feature in enumerate(selected_features):
    axes[ind].plot(feature_ts[:, feature], lw=0.5, label=f"feature {feature}", alpha = 0.9)
    axes[ind].set_xlim(0, feature_ts.shape[0])
    axes[ind].set_ylim([0, 7]) # just to look nice
    axes[ind].set_yticks([0, 5])
    axes[ind].legend()
plt.show()

# %%
def find_relevant_gb_annotations(records, window_start, window_size, 
                                valid_features={'CDS', 'gene', 'mobile_element', 'misc_feature', 
                                              'rRNA', 'tRNA', 'ncRNA', 'Regulatory', 'tmRNA'},
                                valid_qualifiers={'gene', 'locus_id', 'product', 'mobile_element_type'}):
    """
    Extract annotations from GenBank records within a specified window.
    
    Args:
        records: List of GenBank records
        window_start: Start position of window (int)
        window_size: Size of window (int)
        valid_features: Set of feature types to include
        valid_qualifiers: Set of qualifiers to extract
    
    Returns:
        List of annotations: [start, end, type, qualifiers_dict]
    """
    window_end = window_start + window_size
    annotations = []
    
    for record in records:
        for feature in record.features:
            # Skip features outside window
            if feature.location.end < window_start or feature.location.start > window_end:
                continue
                
            if feature.type in valid_features:
                # Calculate relative positions within window
                start = max(0, feature.location.start - window_start)
                end = min(window_size, feature.location.end - window_start)
                
                # Extract relevant qualifiers
                qualifiers = {q: feature.qualifiers[q] for q in valid_qualifiers 
                            if q in feature.qualifiers}
                
                annotations.append([start, end, feature.type, qualifiers])
    
    return annotations


def extract_sequence(genbank_file, start, end, strand="forward"):
    """
    Extract sequence from GenBank file at specific coordinates.
    
    Args:
        genbank_file: Path to GenBank file
        start: Start position (1-based indexing)
        end: End position (1-based indexing)
        strand: "forward" or "complement"
    
    Returns:
        Extracted sequence as string
    """
    record = SeqIO.read(genbank_file, "genbank")
    seq = record.seq[start-1:end]  # Convert to 0-based indexing
    
    if strand.lower() == "complement":
        seq = seq.reverse_complement()
        
    return str(seq)

# Annotation colors
ANNOTATION_COLORS = {
    'CDS': 'white',
    'gene': 'gray', 
    'mobile_element': 'green',
    'misc_feature': 'yellow',
    'rRNA': '#7AC8AC',
    'tRNA': '#662D91',
    'ncRNA': 'white',
    'Regulatory': 'red',
    'tmRNA': 'red'
}

# Get features and plot over a 100kb chunk of the E. coli str. K-12 substr. MG1655 genome, recreating part of the main and supplementary figures
# Download from NCBI: https://www.ncbi.nlm.nih.gov/nuccore/556503834
genbank_file_path = './NC_000913.gb'
start_pos = 4130000
end_pos = 4230000
selected_features = [13606, 26069, 30262, 2812, 15680, 11734, 24568, 15481]

# Load GenBank and get features
records = list(SeqIO.parse(genbank_file_path, "genbank"))
sequence = extract_sequence(genbank_file_path, start_pos, end_pos)
annotations = find_relevant_gb_annotations(records, start_pos, end_pos - start_pos)
feature_ts = get_feature_ts(topk_sae, sequence)

# Plot selected features with genbank annotations visualized as well
fig, axes = plt.subplots(len(selected_features), 1,  figsize=(40, len(selected_features)), sharex=True)
for i, feature_id in enumerate(selected_features):
    axes[i].plot(feature_ts[:, feature_id], lw=0.5, label=f"feature {feature_id}", alpha=0.9)
    for start, end, feature_type, _ in annotations:
        axes[i].axvspan(start, end, color=ANNOTATION_COLORS.get(feature_type, 'black'), alpha=0.2)
    axes[i].set_xlim(0, feature_ts.shape[0])
    axes[i].set_yticks([0, 5])
    axes[i].legend()
plt.show()

# %%