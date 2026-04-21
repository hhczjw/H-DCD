import torch
import torch.nn as nn
import sys
import os
import unittest

# Add project root to sys.path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'models'))

# Import modules
try:
    from models.hmnf_block import HMNFBlock
    from models.hmnf import CoupledHMNF
    from models.hmpn import HMPN
    from models.decouple_encoder import DecoupleEncoder
    from models.feature_projection import TextProjection, AudioVideoProjection
    from models.h_dcd import H_DCD
except ImportError as e:
    print(f"Import Error: {e}")
    # Try adjusting path if running from different location
    sys.path.insert(0, os.path.abspath('H-DCD'))
    sys.path.insert(0, os.path.abspath('H-DCD/models'))
    from models.hmnf_block import HMNFBlock
    from models.hmnf import CoupledHMNF
    from models.hmpn import HMPN
    from models.decouple_encoder import DecoupleEncoder
    from models.feature_projection import TextProjection, AudioVideoProjection
    from models.h_dcd import H_DCD

class TestModules(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.B = 2
        self.L = 32
        self.D = 64
        
    def test_hmnf_block(self):
        print("\nTesting HMNFBlock...")
        # Use headdim=16 to ensure d_in_proj is multiple of 8 for D=64
        block = HMNFBlock(d_model=self.D, headdim=16).to(self.device)
        x = torch.randn(self.B, self.L, self.D).to(self.device)
        fwd_ctx = torch.randn(self.B, self.L, self.D).to(self.device)
        bwd_ctx = torch.randn(self.B, self.L, self.D).to(self.device)
        
        try:
            out = block(x, fwd_context=fwd_ctx, bwd_context=bwd_ctx)
            self.assertEqual(out.shape, (self.B, self.L, self.D))
            print("HMNFBlock Passed")
        except Exception as e:
            self.fail(f"HMNFBlock Failed: {e}")

    def test_coupled_hmnf(self):
        print("\nTesting CoupledHMNF...")
        # Use headdim=16
        model = CoupledHMNF(d_model=self.D, num_layers=2, headdim=16).to(self.device)
        x_a = torch.randn(self.B, self.L, self.D).to(self.device)
        x_v = torch.randn(self.B, self.L, self.D).to(self.device)
        x_l = torch.randn(self.B, self.L, self.D).to(self.device)
        
        try:
            out_a, out_v, out_l = model(x_a, x_v, x_l)
            self.assertEqual(out_a.shape, (self.B, self.L, self.D))
            self.assertEqual(out_v.shape, (self.B, self.L, self.D))
            self.assertEqual(out_l.shape, (self.B, self.L, self.D))
            print("CoupledHMNF Passed")
        except Exception as e:
            self.fail(f"CoupledHMNF Failed: {e}")

    def test_hmpn(self):
        print("\nTesting HMPN...")
        # HMPN uses Mamba2 internally too?
        # Let's check HMPN implementation. Assuming it uses Mamba2 if available.
        # We need to ensure headdim is compatible if it exposes it.
        # HMPN init: d_model, d_state, d_conv, expand, headdim, ngroups, num_heads
        model = HMPN(d_model=self.D, headdim=16).to(self.device)
        x_a = torch.randn(self.B, self.L, self.D).to(self.device)
        x_v = torch.randn(self.B, self.L, self.D).to(self.device)
        x_l = torch.randn(self.B, self.L, self.D).to(self.device)
        
        try:
            out = model(x_a, x_v, x_l)
            # HMPN returns pooled output (B, D)
            self.assertEqual(out.shape, (self.B, self.D))
            print("HMPN Passed")
        except Exception as e:
            self.fail(f"HMPN Failed: {e}")

    def test_decouple_encoder(self):
        print("\nTesting DecoupleEncoder...")
        model = DecoupleEncoder(d_model=self.D, num_modalities=3).to(self.device)
        x_a = torch.randn(self.B, self.L, self.D).to(self.device)
        x_v = torch.randn(self.B, self.L, self.D).to(self.device)
        x_l = torch.randn(self.B, self.L, self.D).to(self.device)
        
        try:
            outputs = model(x_a, x_v, x_l)
            self.assertIsInstance(outputs, dict)
            self.assertIn('s_text', outputs)
            self.assertIn('c_text', outputs)
            self.assertEqual(outputs['s_text'].shape, (self.B, self.L, self.D))
            print("DecoupleEncoder Passed")
        except Exception as e:
            self.fail(f"DecoupleEncoder Failed: {e}")

    def test_feature_projection(self):
        print("\nTesting FeatureProjection...")
        proj_t = TextProjection(input_dim=768, output_dim=self.D).to(self.device)
        proj_av = AudioVideoProjection(input_dim=74, output_dim=self.D).to(self.device)
        
        x_t = torch.randn(self.B, self.L, 768).to(self.device)
        x_a = torch.randn(self.B, self.L, 74).to(self.device)
        
        try:
            out_t = proj_t(x_t)
            out_a = proj_av(x_a)
            self.assertEqual(out_t.shape, (self.B, self.L, self.D))
            self.assertEqual(out_a.shape, (self.B, self.L, self.D))
            print("FeatureProjection Passed")
        except Exception as e:
            self.fail(f"FeatureProjection Failed: {e}")

    def test_h_dcd_integration(self):
        print("\nTesting H_DCD Integration...")
        # Need to ensure H_DCD uses compatible headdim internally
        # H_DCD init doesn't expose headdim directly for HMNF/HMPN, it uses defaults or hardcoded?
        # I need to check H_DCD init again.
        
        model = H_DCD(
            d_model=128, # Use 128 as in default
            hmnf_num_layers=2
        ).to(self.device)
        
        # Mock inputs
        x_t = torch.randn(self.B, self.L, 768).to(self.device)
        x_a = torch.randn(self.B, self.L, 74).to(self.device)
        x_v = torch.randn(self.B, self.L, 35).to(self.device)
        
        try:
            outputs = model(x_t, x_a, x_v)
            self.assertIsInstance(outputs, dict)
            self.assertIn('logits_multi', outputs)
            print("H_DCD Integration Passed")
        except Exception as e:
            self.fail(f"H_DCD Integration Failed: {e}")

if __name__ == '__main__':
    unittest.main()
