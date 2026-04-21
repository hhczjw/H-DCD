"""
Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model
=========================================================================

This module implements the Coupled Mamba architecture for multi-modal sequence modeling.
It features an adaptive state coupling mechanism where the hidden state evolution of one
modality is dynamically influenced by the historical states of other modalities.

Classes:
    - CoupledMambaCell: A single time-step SSM unit with support for external state coupling.
    - CoupledMamba: The main module processing sequences with adaptive coupling.

Dependencies:
    - torch
    - mamba_ssm (Optional, for high-performance parallel mode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

# Try to import Mamba2 for the parallel implementation
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class CoupledMambaCell(nn.Module):
    """
    A single time-step State Space Model (SSM) cell.
    
    Mathematically approximates the discretized SSM update:
        h_t = A * h_{t-1} + B * x_t + Coupling_Term
        y_t = C * h_t
    
    In this simplified cell implementation (for the loop mode), we use 
    Linear layers to approximate the A, B, and C matrices.
    """
    def __init__(self, d_model: int, d_state: int):
        """
        Args:
            d_model (int): The dimension of the input/output features.
            d_state (int): The dimension of the hidden state.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # B matrix: Input projection
        self.x_proj = nn.Linear(d_model, d_state)
        
        # A matrix: State transition (simplified as Linear for demonstration)
        self.h_proj = nn.Linear(d_state, d_state)
        
        # C matrix: Output projection
        self.out_proj = nn.Linear(d_state, d_model)
        
        # Nonlinearity (Mamba uses SiLU/Swish)
        self.act = nn.SiLU()

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, coupled_influence: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.

        Args:
            x_t (Tensor): Current input [Batch, d_model]
            h_prev (Tensor): Previous state [Batch, d_state]
            coupled_influence (Tensor, optional): Weighted sum of other modalities' states [Batch, d_state]

        Returns:
            y_t (Tensor): Current output [Batch, d_model]
            h_t (Tensor): Updated state [Batch, d_state]
        """
        # 1. Basic SSM Update: h = A*h_prev + B*x
        # Note: In real Mamba, A is usually diagonal and parameter-efficient. 
        # Here we use a full Linear for the logic demonstration.
        state_update = self.h_proj(h_prev) + self.x_proj(x_t)
        
        # 2. Add Coupling Influence
        if coupled_influence is not None:
            state_update = state_update + coupled_influence
            
        # 3. Activation and Update
        h_t = self.act(state_update)
        
        # 4. Output Projection
        y_t = self.out_proj(h_t)
        
        return y_t, h_t


class CoupledMamba(nn.Module):
    """
    Coupled Mamba Module for Multi-modal Fusion.
    
    Features:
    - Independent SSM channels for Audio, Visual, and Lexical modalities.
    - Adaptive Weighting Mechanism: Dynamically learns the importance of cross-modal history.
    - Dual Implementation: 
        1. `forward_loop`: Explicit Python loop for exact state coupling logic (slow, educational).
        2. `forward_parallel`: High-performance approximation using concatenated Mamba (fast).
    """
    def __init__(self, d_model: int, d_state: int = 64, use_parallel: bool = False):
        """
        Args:
            d_model (int): Feature dimension per modality.
            d_state (int): SSM state dimension.
            use_parallel (bool): If True, uses the fast Mamba2 implementation (requires mamba_ssm).
                                 If False, uses the explicit loop implementation.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.use_parallel = use_parallel
        self.modalities = ['audio', 'visual', 'lexical']
        self.num_modalities = len(self.modalities)

        # ====================================================================
        # Components for Loop Implementation (Explicit Coupling)
        # ====================================================================
        # 1. Independent Mamba Cores
        self.mamba_cores = nn.ModuleDict({
            m: CoupledMambaCell(d_model, d_state) for m in self.modalities
        })

        # 2. Coupling Projections (State Space Mapping)
        # Maps state from src modality to tgt modality
        self.coupling_projections = nn.ModuleDict()
        for tgt in self.modalities:
            for src in self.modalities:
                if src == tgt: continue
                layer_name = f"{src}_to_{tgt}"
                self.coupling_projections[layer_name] = nn.Linear(d_state, d_state, bias=False)

        # 3. Adaptive Weight Network
        # Input: Concatenated states of all modalities (3 * d_state)
        # Output: Attention weights for 3x3 interactions
        self.weight_net = nn.Sequential(
            nn.Linear(self.num_modalities * d_state, d_state),
            nn.Tanh(),
            nn.Linear(d_state, self.num_modalities * self.num_modalities)
        )

        self.layer_norms = nn.ModuleDict({
            m: nn.LayerNorm(d_model) for m in self.modalities
        })

        # ====================================================================
        # Components for Parallel Implementation (Implicit Coupling)
        # ====================================================================
        if MAMBA_AVAILABLE:
            # We treat the 3 modalities as a single wide sequence or channel-concatenated input
            # This allows mixing via the input projections and convolutions of Mamba2
            self.parallel_mamba = Mamba2(
                d_model=d_model * 3, # Process all modalities together
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.parallel_proj = nn.ModuleDict({
                m: nn.Linear(d_model * 3, d_model) for m in self.modalities
            })

    def forward(self, x_audio: torch.Tensor, x_visual: torch.Tensor, x_lexical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_audio:   [Batch, Seq_Len, d_model]
            x_visual:  [Batch, Seq_Len, d_model]
            x_lexical: [Batch, Seq_Len, d_model]
            
        Returns:
            out_audio, out_visual, out_lexical: Processed features with same shape as input.
        """
        if self.use_parallel and MAMBA_AVAILABLE:
            return self.forward_parallel(x_audio, x_visual, x_lexical)
        else:
            if self.use_parallel and not MAMBA_AVAILABLE:
                print("Warning: Mamba2 not available. Falling back to loop implementation.")
            return self.forward_loop(x_audio, x_visual, x_lexical)

    def forward_loop(self, x_audio, x_visual, x_lexical):
        """
        Explicit step-wise implementation with Adaptive State Coupling.
        This faithfully implements the Coupled Mamba logic where h_t depends on h_{t-1}^{other}.
        """
        batch_size, seq_len, _ = x_audio.shape
        device = x_audio.device

        # Initialize states
        h_states = {
            m: torch.zeros(batch_size, self.d_state, device=device) 
            for m in self.modalities
        }
        
        outputs = {m: [] for m in self.modalities}

        # --- Time Step Loop ---
        for t in range(seq_len):
            # Snapshot previous states
            h_prev = {k: v.clone() for k, v in h_states.items()}
            
            # 1. Adaptive Weight Generation
            # Concatenate all previous states: [Batch, 3 * d_state]
            h_concat = torch.cat([h_prev['audio'], h_prev['visual'], h_prev['lexical']], dim=-1)
            
            # Generate weights: [Batch, 3, 3]
            # raw_weights[b, i, j] -> importance of src j for tgt i
            raw_weights = self.weight_net(h_concat).view(batch_size, self.num_modalities, self.num_modalities)
            attn_weights = F.softmax(raw_weights, dim=-1) 
            
            # 2. Per-Modality Update
            for tgt_idx, tgt_modality in enumerate(self.modalities):
                # Get current input
                if tgt_modality == 'audio': x_t = x_audio[:, t, :]
                elif tgt_modality == 'visual': x_t = x_visual[:, t, :]
                else: x_t = x_lexical[:, t, :]

                # Calculate Weighted Coupled Context
                coupling_context = 0.0
                
                for src_idx, src_modality in enumerate(self.modalities):
                    # Self-loop is handled internally by the cell, so we skip adding it to coupling_context
                    # However, the weight w_ii still matters as it scales the other weights via Softmax
                    if src_modality == tgt_modality:
                        continue
                    
                    # Get adaptive weight w_{ij}
                    w_ij = attn_weights[:, tgt_idx, src_idx].unsqueeze(-1) # [Batch, 1]
                    
                    # Project source state to target space
                    proj_layer = self.coupling_projections[f"{src_modality}_to_{tgt_modality}"]
                    h_src_projected = proj_layer(h_prev[src_modality])
                    
                    # Accumulate
                    coupling_context = coupling_context + (w_ij * h_src_projected)
                
                # Update Mamba Cell
                y_t, h_new = self.mamba_cores[tgt_modality](
                    x_t, 
                    h_prev[tgt_modality], 
                    coupled_influence=coupling_context
                )
                
                h_states[tgt_modality] = h_new
                outputs[tgt_modality].append(y_t)

        # Stack outputs
        out_audio = torch.stack(outputs['audio'], dim=1)
        out_visual = torch.stack(outputs['visual'], dim=1)
        out_lexical = torch.stack(outputs['lexical'], dim=1)

        # Residual + Norm
        out_audio = self.layer_norms['audio'](out_audio + x_audio)
        out_visual = self.layer_norms['visual'](out_visual + x_visual)
        out_lexical = self.layer_norms['lexical'](out_lexical + x_lexical)

        return out_audio, out_visual, out_lexical

    def forward_parallel(self, x_audio, x_visual, x_lexical):
        """
        High-performance approximation using Mamba2.
        
        Instead of explicit state-level coupling (which requires custom CUDA kernels for 
        dense state dependencies), we concatenate the modalities into a single wide sequence.
        The Mamba2 block then processes them together, allowing mixing via its internal
        projections and convolutions.
        """
        # Concatenate along feature dimension: [Batch, Seq_Len, 3*d_model]
        x_concat = torch.cat([x_audio, x_visual, x_lexical], dim=-1)
        
        # Pass through Mamba2
        # Mamba2 mixes channels via in_proj and conv1d before the SSM scan
        out_concat = self.parallel_mamba(x_concat)
        
        # Split back to modalities (using separate projections to disentangle)
        out_audio = self.parallel_proj['audio'](out_concat)
        out_visual = self.parallel_proj['visual'](out_concat)
        out_lexical = self.parallel_proj['lexical'](out_concat)
        
        # Residual + Norm
        out_audio = self.layer_norms['audio'](out_audio + x_audio)
        out_visual = self.layer_norms['visual'](out_visual + x_visual)
        out_lexical = self.layer_norms['lexical'](out_lexical + x_lexical)
        
        return out_audio, out_visual, out_lexical


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("Coupled Mamba Module Test")
    print("="*60)
    
    # Configuration
    BATCH_SIZE = 4
    SEQ_LEN = 32
    D_MODEL = 64
    D_STATE = 32
    
    # Instantiate Model
    # Note: Set use_parallel=True to test Mamba2 if installed
    model = CoupledMamba(d_model=D_MODEL, d_state=D_STATE, use_parallel=False)
    
    # Create Dummy Inputs
    x_a = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    x_v = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    x_l = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    
    print(f"Input Shapes: {x_a.shape}")
    print(f"Mode: {'Parallel (Mamba2)' if model.use_parallel else 'Loop (Explicit Coupling)'}")
    
    # Forward Pass
    out_a, out_v, out_l = model(x_a, x_v, x_l)
    
    print("\nForward Pass Successful!")
    print(f"Output Audio:   {out_a.shape}")
    print(f"Output Visual:  {out_v.shape}")
    print(f"Output Lexical: {out_l.shape}")
    
    # Check for NaNs
    if torch.isnan(out_a).any():
        print("Warning: NaNs detected in output!")
    else:
        print("Output numerical check passed.")
    print("="*60)
