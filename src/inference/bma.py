import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

class MixtureOfCausalLM(PreTrainedModel):
    def __init__(self, model1, model2, model2_weight=1.0, window_size=None):
        super().__init__(model1.config)
        self.model1 = model1
        self.model2 = model2
        self.log_model2_weight = np.log(model2_weight)
        self.window_size = window_size  # None means use all tokens (current behavior)
        self._reset_state()

    def _reset_state(self):
        self.running_logprobs1 = None
        self.running_logprobs2 = None
        self.past1 = None
        self.past2 = None
        # For windowed approach
        self.logprob_history1 = []
        self.logprob_history2 = []

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model1.to(*args, **kwargs)
        self.model2.to(*args, **kwargs)
        self._reset_state()
        return self

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, return_alphas=False, eos_token_id=None, init_with_prompt=True, window_size=None):
        self._reset_state()
        self.eval()
        
        # Use instance window_size if not provided
        if window_size is None:
            window_size = self.window_size
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        with torch.no_grad():
            # Compute initial alpha based on prompt likelihood
            out1 = self.model1(input_ids, use_cache=True)
            out2 = self.model2(input_ids, use_cache=True)
            self.past1 = out1.past_key_values
            self.past2 = out2.past_key_values
            
            if input_ids.size(1) > 1 and init_with_prompt:
                # Evaluate models on the prompt (input_ids[:-1] -> input_ids[1:])
                logits1 = out1.logits[:, :-1, :]  # All but last position
                logits2 = out2.logits[:, :-1, :]
                targets = input_ids[:, 1:]  # Shifted targets
                
                log_probs1 = F.log_softmax(logits1, dim=-1)
                log_probs2 = F.log_softmax(logits2, dim=-1)
                
                # Get per-token log probabilities
                token_logp1 = log_probs1.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
                token_logp2 = log_probs2.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
                
                if window_size is None:
                    # Use all prompt tokens
                    prompt_logp1 = token_logp1.sum(dim=1)
                    prompt_logp2 = token_logp2.sum(dim=1)
                    self.running_logprobs1 = prompt_logp1
                    self.running_logprobs2 = prompt_logp2
                else:
                    # Initialize windowed history with prompt tokens
                    for i in range(token_logp1.size(1)):
                        self.logprob_history1.append(token_logp1[:, i])
                        self.logprob_history2.append(token_logp2[:, i])
                    
                    # Keep only the last window_size tokens
                    if len(self.logprob_history1) > window_size:
                        self.logprob_history1 = self.logprob_history1[-window_size:]
                        self.logprob_history2 = self.logprob_history2[-window_size:]
                    
                    # Sum the windowed history
                    self.running_logprobs1 = torch.stack(self.logprob_history1).sum(dim=0)
                    self.running_logprobs2 = torch.stack(self.logprob_history2).sum(dim=0)
            else:
                # Single token prompt - start neutral
                self.running_logprobs1 = torch.zeros(batch_size, device=device)
                self.running_logprobs2 = torch.zeros(batch_size, device=device)
                self.logprob_history1 = []
                self.logprob_history2 = []

            print("Initial log probs for model 1:", self.running_logprobs1)
            print("Initial log probs for model 2:", self.running_logprobs2)

            alpha_history = []
            generated = input_ids.clone()
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            lengths = torch.full((batch_size,), max_length + input_ids.size(1), dtype=torch.long, device=device)
            
            for step in range(max_length):
                last_token = generated[:, -1:]
                
                out1 = self.model1(last_token, past_key_values=self.past1, use_cache=True)
                out2 = self.model2(last_token, past_key_values=self.past2, use_cache=True)
                
                logits1 = out1.logits[:, -1, :]
                logits2 = out2.logits[:, -1, :]
                
                self.past1 = out1.past_key_values
                self.past2 = out2.past_key_values
                
                # Compute mixture weights based on running log probs
                log_weights = torch.stack([self.running_logprobs1, self.log_model2_weight + self.running_logprobs2], dim=0)
                weights = F.softmax(log_weights, dim=0)
                alpha = weights[0].unsqueeze(1)  # shape: (batch, 1)
                
                alpha_history.append(alpha.detach())
                
                # Compute mixture
                probs1 = F.softmax(logits1, dim=-1)
                probs2 = F.softmax(logits2, dim=-1)
                mixed_probs = alpha * probs1 + (1 - alpha) * probs2
                mixed_logits = torch.log(mixed_probs + 1e-8)
                
                # Apply temperature and top-k
                if temperature != 1.0:
                    mixed_logits = mixed_logits / temperature
                    
                if top_k is not None and top_k > 0:
                    top_k_val = min(top_k, mixed_logits.size(-1))
                    values, indices = torch.topk(mixed_logits, top_k_val)
                    probs = torch.full_like(mixed_logits, float('-inf'))
                    probs.scatter_(1, indices, values)
                    mixed_logits = probs
                
                # Sample next token
                probs = F.softmax(mixed_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Early stopping logic...
                if eos_token_id is not None:
                    just_finished = (next_token.squeeze(-1) == eos_token_id) & (~finished)
                    lengths[just_finished] = generated.size(1) + 1
                    finished = finished | just_finished
                    next_token = torch.where(
                        finished.unsqueeze(-1),
                        torch.full_like(next_token, eos_token_id),
                        next_token
                    )
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Update running log probs with the token we just selected
                token_id = next_token.squeeze(-1)
                logp1 = F.log_softmax(logits1, dim=-1).gather(1, token_id.unsqueeze(1)).squeeze(1)
                logp2 = F.log_softmax(logits2, dim=-1).gather(1, token_id.unsqueeze(1)).squeeze(1)

                if window_size is None:
                    # Cumulative (original behavior)
                    self.running_logprobs1 += logp1
                    self.running_logprobs2 += logp2
                else:
                    # Windowed approach - compute joint probability of last K tokens
                    self.logprob_history1.append(logp1)
                    self.logprob_history2.append(logp2)
                    
                    # Keep only the last window_size tokens
                    if len(self.logprob_history1) > window_size:
                        self.logprob_history1.pop(0)
                        self.logprob_history2.pop(0)
                    
                    # Compute joint log probability of the window using chain rule
                    # p(x_{t-k+1}, ..., x_t) = p(x_{t-k+1}|context) * p(x_{t-k+2}|context,x_{t-k+1}) * ... * p(x_t|context,...,x_{t-1})
                    
                    if len(self.logprob_history1) == 1:
                        # Only one token in window - use its conditional probability
                        self.running_logprobs1 = self.logprob_history1[0]
                        self.running_logprobs2 = self.logprob_history2[0]
                    else:
                        # Multiple tokens - need to recompute first token's probability given its actual context
                        window_len = len(self.logprob_history1)
                        window_start_idx = generated.size(1) - window_len
                        
                        # Get context before the window starts
                        if window_start_idx > 0:
                            context = generated[:, :window_start_idx]
                            first_window_token = generated[:, window_start_idx]
                            
                            # Compute p(first_window_token | context) by re-running models on context
                            with torch.no_grad():
                                out1_ctx = self.model1(context)
                                out2_ctx = self.model2(context)
                                
                                first_token_logits1 = out1_ctx.logits[:, -1, :]
                                first_token_logits2 = out2_ctx.logits[:, -1, :]
                                
                                first_logp1 = F.log_softmax(first_token_logits1, dim=-1).gather(1, first_window_token.unsqueeze(1)).squeeze(1)
                                first_logp2 = F.log_softmax(first_token_logits2, dim=-1).gather(1, first_window_token.unsqueeze(1)).squeeze(1)
                        else:
                            # Window starts at the very beginning - use uniform prior (log prob = 0)
                            # Or you could use a learned unconditional probability here
                            first_logp1 = torch.zeros_like(self.logprob_history1[0])
                            first_logp2 = torch.zeros_like(self.logprob_history2[0])
                        
                        # Joint probability = p(first|context) + sum of remaining conditional probabilities
                        # The remaining probabilities in logprob_history are already conditional on their proper contexts
                        remaining_logp1 = torch.stack(self.logprob_history1[1:]).sum(dim=0) if len(self.logprob_history1) > 1 else torch.zeros_like(first_logp1)
                        remaining_logp2 = torch.stack(self.logprob_history2[1:]).sum(dim=0) if len(self.logprob_history2) > 1 else torch.zeros_like(first_logp2)
                        
                        self.running_logprobs1 = first_logp1 + remaining_logp1
                        self.running_logprobs2 = first_logp2 + remaining_logp2

                if finished.all():
                    break

            # Truncation logic...
            max_gen_len = generated.size(1)
            output = []
            for i in range(batch_size):
                end = lengths[i].item()
                output.append(generated[i, :end])
            padded = torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=eos_token_id if eos_token_id is not None else 0)

        if return_alphas:
            alphas = torch.cat(alpha_history, dim=1).squeeze(-1)
            return padded, alphas
        else:
            return padded