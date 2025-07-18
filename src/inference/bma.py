import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer

class MixtureOfCausalLM(PreTrainedModel):
    def __init__(self, model1, model2):
        super().__init__(model1.config)
        self.model1 = model1
        self.model2 = model2
        self._reset_state()

    def _reset_state(self):
        self.running_logprobs1 = None
        self.running_logprobs2 = None
        self.past1 = None
        self.past2 = None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model1.to(*args, **kwargs)
        self.model2.to(*args, **kwargs)
        self._reset_state()
        return self

    def generate(self, input_ids, max_length=50, temperature=1.0, top_k=50, return_alphas=False, eos_token_id=None):
        self._reset_state()
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        self.running_logprobs1 = torch.zeros(batch_size, device=device)
        self.running_logprobs2 = torch.zeros(batch_size, device=device)
        
        alpha_history = []
        generated = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        lengths = torch.full((batch_size,), max_length + input_ids.size(1), dtype=torch.long, device=device)
        
        # Process initial context for both models
        with torch.no_grad():
            out1 = self.model1(input_ids, use_cache=True)
            out2 = self.model2(input_ids, use_cache=True)
            self.past1 = out1.past_key_values
            self.past2 = out2.past_key_values

        for step in range(max_length):
            last_token = generated[:, -1:]
            
            out1 = self.model1(last_token, past_key_values=self.past1, use_cache=True)
            out2 = self.model2(last_token, past_key_values=self.past2, use_cache=True)
            
            logits1 = out1.logits[:, -1, :]
            logits2 = out2.logits[:, -1, :]
            
            self.past1 = out1.past_key_values
            self.past2 = out2.past_key_values
            
            # Compute mixture weights based on running log probs
            if step == 0:
                alpha = torch.ones(batch_size, 1, device=device) * 0.5
            else:
                log_alpha = self.running_logprobs1
                log_beta = self.running_logprobs2
                alpha = torch.exp(log_alpha - torch.logsumexp(torch.stack([log_alpha, log_beta]), dim=0)).unsqueeze(1)
            
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
            
            # Early stopping: if eos_token_id is generated, mark as finished
            if eos_token_id is not None:
                just_finished = (next_token.squeeze(-1) == eos_token_id) & (~finished)
                lengths[just_finished] = generated.size(1) + 1  # +1 for the new token
                finished = finished | just_finished
            
            # For finished sequences, keep appending pad tokens (or eos_token_id)
            if eos_token_id is not None:
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
            
            self.running_logprobs1 += logp1
            self.running_logprobs2 += logp2

            # If all sequences are finished, break early
            if finished.all():
                break

        # Truncate each sequence at its first eos (or max length)
        max_gen_len = generated.size(1)
        output = []
        for i in range(batch_size):
            end = lengths[i].item()
            output.append(generated[i, :end])
        # Pad to the same length
        padded = torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=eos_token_id if eos_token_id is not None else 0)

        if return_alphas:
            alphas = torch.cat(alpha_history, dim=1).squeeze(-1)
            return padded, alphas
        else:
            return padded