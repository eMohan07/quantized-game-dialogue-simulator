"""Your code. One stub per milestone; build them in order and keep each one small."""
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq

# Default game dialogue dataset featuring classic RPG NPC interactions and quest lines
DEFAULT_GAME_DIALOGUE = """
[Hero] The kingdom is in peril. We must find the ancient relic before dusk.
[Elder] The path ahead is treacherous, young traveler. Take this warding charm.
[Hero] Thank you, Elder. What perils await within the Whispering Caverns?
[Elder] Dark wraiths and forgotten guardians lurk in the subterranean mist.
[Merchant] Welcome, traveler! Looking to buy rare potions or sharp steel?
[Hero] How much for a flask of sunfire elixir?
[Merchant] Fifty gold pieces, no less! Quality like this does not come cheap.
[Guard] Halt! State your business in the Royal Citadel!
[Hero] I carry an urgent message for the King from the southern outpost.
[Guard] Very well, pass through the iron gates, but keep your blade sheathed.
[Wizard] The arcane leylines are destabilizing. Can you not feel the hum in the stones?
[Hero] Is there a way to calm the elemental surge?
[Wizard] You must place three celestial crystals upon the high altar at Midnight Peak.
[Rogue] Keep your voice down. The shadows hide more than just cutpurses tonight.
[Hero] Did you spot the goblin scouting party near the watchtower?
[Rogue] Aye, five of them, armed with poisoned daggers. We strike from above.
[Goblin] Shiny trinkets! Leave all your shiny gold or taste rusted steel!
[Hero] Back off, foul creature, before my sword decides your fate!
[Companion] My bow is notched and ready, friend. We stand united against the darkness.
[Hero] Together, we shall restore peace to the realm.
[Elder] Remember, strength alone will not shatter the curse; wisdom must guide your hand.
[Merchant] Safe travels, adventurer! Return with monster trophies and I will pay handsomely!
[Guard] Watch the perimeter! Torches lit until dawn!
""".strip()


def clean_dialogue(text: str) -> str:
    """Clean and normalize game dialogue text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_lines = [line for line in lines if line]
    return "\n".join(cleaned_lines)


class CharTokenizer:
    """Character-level tokenizer for small dialogue models."""

    def __init__(self, corpus: str):
        self.chars = sorted(list(set(corpus)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """Encode text string into token IDs."""
        return [self.char_to_ix.get(ch, 0) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back into string."""
        return "".join([self.ix_to_char.get(int(idx), "") for idx in tokens])


# --- Milestone 1: Prepare a small game dialogue text corpus.
# Hint: Focus on cleaning and tokenizing your game dialogue text data.
def milestone_1(raw_corpus: Optional[str] = None) -> Dict[str, Any]:
    """Prepare, clean, and tokenize a game dialogue corpus."""
    corpus_text = raw_corpus if raw_corpus is not None else DEFAULT_GAME_DIALOGUE
    cleaned = clean_dialogue(corpus_text)
    tokenizer = CharTokenizer(cleaned)
    tokens = tokenizer.encode(cleaned)
    return {
        "corpus": cleaned,
        "tokenizer": tokenizer,
        "tokens": tokens,
        "vocab_size": tokenizer.vocab_size,
    }


# --- GPT Architecture Components ---

class MiniGPTConfig:
    """Configuration for small in-game dialogue GPT model."""

    def __init__(
        self,
        vocab_size: int = 65,
        block_size: int = 32,
        n_embd: int = 48,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.0,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


class QuantLinear(nn.Module):
    """Linear layer wrapped with QuantStub and DeQuantStub for quantization compatibility."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.quant = tq.QuantStub()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequant(self.linear(self.quant(x)))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = QuantLinear(config.n_embd, 3 * config.n_embd)
        self.c_proj = QuantLinear(config.n_embd, config.n_embd)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class FeedForward(nn.Module):
    """Transformer MLP feed-forward block."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.fc1 = QuantLinear(config.n_embd, 4 * config.n_embd)
        self.act = nn.ReLU()
        self.fc2 = QuantLinear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer block with pre-LayerNorm and residual connections."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Compact GPT model designed for dialogue generation in games."""

    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = QuantLinear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence length {T} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(
    tokens: List[int], block_size: int, batch_size: int, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous sequences from token list."""
    data = torch.tensor(tokens, dtype=torch.long)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


def train_gpt(
    model: MiniGPT,
    tokens: List[int],
    steps: int = 100,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train MiniGPT on dialogue tokens."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    losses = []

    for step in range(steps):
        x, y = get_batch(tokens, model.config.block_size, batch_size, device=device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    model.eval()
    return {
        "losses": losses,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }


# --- Milestone 2: Implement a basic GPT architecture and train it on the corpus.
# Hint: Use the 'Build GPT from Scratch' drill's techniques to define and train your model.
def milestone_2(
    corpus_data: Optional[Dict[str, Any]] = None,
    steps: int = 80,
    lr: float = 1e-3,
    batch_size: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Define a basic GPT model and train it on the dialogue corpus."""
    torch.manual_seed(seed)
    random.seed(seed)
    if corpus_data is None:
        corpus_data = milestone_1()

    config = MiniGPTConfig(
        vocab_size=corpus_data["vocab_size"],
        block_size=32,
        n_embd=48,
        n_head=4,
        n_layer=2,
    )
    model = MiniGPT(config)
    train_results = train_gpt(
        model,
        corpus_data["tokens"],
        steps=steps,
        batch_size=batch_size,
        lr=lr,
    )
    return {
        "model": model,
        "config": config,
        "tokenizer": corpus_data["tokenizer"],
        "initial_loss": train_results["initial_loss"],
        "final_loss": train_results["final_loss"],
        "losses": train_results["losses"],
    }


def generate_dialogue(
    model: nn.Module,
    tokenizer: CharTokenizer,
    prompt: str = "[Hero] ",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: Optional[int] = 5,
    device: str = "cpu",
) -> str:
    """Generate game dialogue given a prompt and trained model."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [0]
    idx = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    block_size = getattr(model, "config", None)
    max_ctx = block_size.block_size if block_size is not None else 32

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_ctx:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

    generated_tokens = idx[0].tolist()
    return tokenizer.decode(generated_tokens)


# --- Milestone 3: Generate sample dialogue from the trained GPT.
# Hint: Implement a text generation function using your trained GPT to produce sample dialogue.
def milestone_3(
    model: Optional[MiniGPT] = None,
    tokenizer: Optional[CharTokenizer] = None,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
) -> Dict[str, Any]:
    """Generate sample dialogue responses for game characters."""
    if model is None or tokenizer is None:
        m2_res = milestone_2(steps=60)
        model = m2_res["model"]
        tokenizer = m2_res["tokenizer"]

    test_prompts = (
        prompts
        if prompts is not None
        else ["[Hero] ", "[Guard] ", "[Merchant] ", "[Wizard] "]
    )
    generated_dialogues = {}
    for p in test_prompts:
        generated_dialogues[p] = generate_dialogue(
            model=model,
            tokenizer=tokenizer,
            prompt=p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    return {
        "generated_samples": generated_dialogues,
        "model": model,
        "tokenizer": tokenizer,
    }


# --- Milestone 4: Apply dynamic and static post-training quantization to the model.
# Hint: Refer to the 'Shrink a Neural Network with Quantisation' drill for applying dynamic and static post-training quantization.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Measure and compare the original and quantized model sizes.
# Hint: Save your original and quantized models and check their file sizes to measure memory footprint.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")


# --- Milestone 6: Simulate inference speed by comparing operations on original versus quantized weights.
# Hint: Measure the time taken for a fixed number of inference steps for both models, as shown in the 'Shrink a Neural Network with Quantisation' drill.
def milestone_6():  # TODO(milestone 6)
    raise NotImplementedError("TODO(milestone 6)")
