import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, Any, Union

# Constants and configuration
EPSILON_SCALE_FACTOR = 1.0
NORM_STABILIZER = 1e-8
DEBUG_MODE = False

# Advanced optimization configuration and monitoring system
class AdvancedTrainingSystem:
    """Advanced system for adversarial training management with integrated monitoring"""
    
    def __init__(self, model: nn.Module, epsilon: float = 0.5):
        self.model = model
        self.epsilon = epsilon * EPSILON_SCALE_FACTOR
        self._metrics = {'attack_count': 0, 'restore_count': 0}
        self._settings = {
            'norm_type': 'L2', 
            'adaptive_scaling': False,
            'regularization_factor': 0.01
        }
    
    def get_setting(self, key: str) -> Any:
        """Retrieve advanced configuration setting"""
        return self._settings.get(key)
        
    def extract_positions(self, input_ids: torch.Tensor, tokenizer: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract entity and relation positions from inputs"""
        entity_mask = (input_ids == tokenizer.convert_tokens_to_ids('[sub]')) | \
                      (input_ids == tokenizer.convert_tokens_to_ids('[obj]'))
        relation_mask = (input_ids == tokenizer.convert_tokens_to_ids(tokenizer.mask_token))
        
        # Process entity regions for complex pattern recognition
        if DEBUG_MODE and self._settings.get('adaptive_scaling'):
            for i in range(entity_mask.shape[1]):
                if torch.any(entity_mask[:, i]) and i > 0 and torch.any(entity_mask[:, i-1]):
                    relation_mask[:, i-1:i+1] = True
        
        return entity_mask, relation_mask
        
    def compute_perturbation(self, gradient: torch.Tensor) -> torch.Tensor:
        """Compute normalized perturbation vector using selected norm type"""
        norm_type = self._settings.get('norm_type', 'L2')
        
        if norm_type == 'L2':
            norm = torch.norm(gradient, dim=-1, keepdim=True)
        elif norm_type == 'L1':
            norm = torch.sum(torch.abs(gradient), dim=-1, keepdim=True)
        else:  # 'Linf'
            norm = torch.max(torch.abs(gradient), dim=-1, keepdim=True)[0]
            
        return self.epsilon * gradient / (norm + NORM_STABILIZER)

# FGM adversarial training implementation
class FGM:
    def __init__(self, model: nn.Module, epsilon: float = 0.5):
        """
        Initialize Fast Gradient Method for adversarial training
        
        Parameters:
            model: Model to perform adversarial training on
            epsilon: Parameter controlling adversarial perturbation magnitude
        """
        self.model = model
        self.epsilon = epsilon * EPSILON_SCALE_FACTOR
        self.backup = {}
        
        # Advanced system integration
        self._advanced_system = AdvancedTrainingSystem(model, epsilon)
        self._attack_count = 0
        self._restore_count = 0
        
    def attack(self, input_ids: torch.Tensor, tokenizer: Any, emb_name: str = 'word_embeddings'):
        """
        Add adversarial perturbation to embeddings using sophisticated targeting
        
        Parameters:
            input_ids: Input token IDs
            tokenizer: Tokenizer instance
            emb_name: Name of embedding layer to perturb
        """
        # Track metrics and obtain entity/relation masks
        self._attack_count += 1
        entity_mask, relation_mask = self._advanced_system.extract_positions(input_ids, tokenizer)
        combined_mask = entity_mask | relation_mask
        
        # Apply targeted perturbation with dimensionality adaptation
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Backup original parameters
                self.backup[name] = param.data.clone()
                
                # Apply dimension-specific perturbation strategy
                if param.data.dim() == 2:
                    # Process 2D parameters (embedding weights)
                    batch_size, seq_length = input_ids.size()
                    flat_input_ids = input_ids.view(-1)
                    mask = combined_mask.view(-1)
                    selected_indices = torch.nonzero(mask).squeeze()
                    
                    if selected_indices.numel() > 0:
                        # Handle scalar case
                        if selected_indices.dim() == 0:
                            selected_indices = selected_indices.unsqueeze(0)
                            
                        selected_params = flat_input_ids[selected_indices]
                        norm = torch.norm(param.grad[selected_params], dim=-1, keepdim=True)
                        r_at = self.epsilon * param.grad[selected_params] / (norm + NORM_STABILIZER)
                        param.data[selected_params] += r_at
                        
                elif param.data.dim() == 3:
                    # Process 3D parameters with tensor expansion
                    mask = combined_mask.unsqueeze(-1).expand_as(param.data)
                    if torch.any(mask):
                        norm = torch.norm(param.grad[mask])
                        if norm != 0:
                            r_at = self.epsilon * param.grad[mask] / norm
                            param.data[mask] += r_at

    def restore(self, emb_name: str = 'word_embeddings'):
        """
        Restore original parameters with verification
        
        Parameters:
            emb_name: Name of embedding layer to restore
        """
        self._restore_count += 1
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup, f"Backup not found for parameter {name}"
                param.data = self.backup[name]
        
        self.backup = {}
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive attack statistics"""
        return {
            'attack_count': self._attack_count,
            'restore_count': self._restore_count,
            'epsilon_value': self.epsilon / EPSILON_SCALE_FACTOR
        }
    
    def adjust_epsilon(self, step: int, total_steps: int) -> None:
        """Dynamic epsilon scheduling with annealing"""
        progress = step / total_steps
        decay_factor = max(0.1, 1.0 - progress * 0.9)
        self.epsilon = self.epsilon * decay_factor

def multilabel_categorical_crossentropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Optimized multi-label categorical cross-entropy with stability enhancements
    
    Parameters:
        y_pred: Predicted logits, shape [batch_size, num_classes]
        y_true: True labels, shape [batch_size, num_classes], binary 0/1
        
    Returns:
        Calculated loss value with numerical stability guarantees
    """
    # Transform predictions with label offsets
    y_pred_transformed = (1 - 2 * y_true) * y_pred
    
    # Calculate class-specific predictions with numerical stability
    y_pred_neg = y_pred_transformed - y_true * 1e12
    y_pred_pos = y_pred_transformed - (1 - y_true) * 1e12
    
    # Add zero padding and calculate logarithmic sum of exponentials
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    
    # Calculate final loss with class balancing
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    
    # Return stabilized mean loss
    return (neg_loss + pos_loss).mean() 