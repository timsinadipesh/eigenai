"""
ONNX Gemma 3n E2B Download Script
Downloads pre-quantized ONNX models for efficient CPU inference
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import argparse

def download_onnx_models(cache_dir=None, quantization="mixed"):
    """Download ONNX Gemma 3n E2B models with specified quantization"""
    
    model_id = "onnx-community/gemma-3n-E2B-it-ONNX"
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "onnx_models" / "gemma-3n-e2b"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading ONNX Gemma 3n E2B models to: {cache_dir}")
    print(f"Quantization strategy: {quantization}")
    print("="*60)
    
    # Define quantization options
    quantization_configs = {
        "mixed": {
            "embed_tokens": "embed_tokens_quantized.onnx",
            "audio_encoder": "audio_encoder.onnx", 
            "vision_encoder": "vision_encoder.onnx",
            "decoder": "decoder_model_merged_q4.onnx"
        },
        "max_savings": {
            "embed_tokens": "embed_tokens_quantized.onnx",
            "audio_encoder": "audio_encoder_quantized.onnx",
            "vision_encoder": "vision_encoder_quantized.onnx", 
            "decoder": "decoder_model_merged_q4.onnx"
        },
        "high_quality": {
            "embed_tokens": "embed_tokens.onnx",
            "audio_encoder": "audio_encoder.onnx",
            "vision_encoder": "vision_encoder.onnx",
            "decoder": "decoder_model_merged.onnx"
        }
    }
    
    if quantization not in quantization_configs:
        quantization = "mixed"
        print(f"Unknown quantization '{quantization}', using 'mixed'")
    
    config = quantization_configs[quantization]
    
    try:
        # Download processor files first (lightweight)
        print("1. Downloading processor files...")
        processor_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
            "preprocessor_config.json"
        ]
        
        for file in processor_files:
            try:
                hf_hub_download(
                    repo_id=model_id,
                    filename=file,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✓ Downloaded {file}")
            except Exception as e:
                print(f"⚠ Could not download {file}: {e}")
        
        print(f"\n2. Downloading ONNX model files ({quantization})...")
        
        # Download ONNX models
        onnx_files = [
            f"onnx/{config['embed_tokens']}",
            f"onnx/{config['audio_encoder']}",
            f"onnx/{config['vision_encoder']}", 
            f"onnx/{config['decoder']}"
        ]
        
        total_size = 0
        for file_path in onnx_files:
            try:
                print(f"Downloading {file_path}...")
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_path,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                
                size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
                total_size += size_mb
                print(f"✓ Downloaded {file_path} ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"✗ Failed to download {file_path}: {e}")
                return False
        
        print(f"\n3. Download Summary:")
        print(f"✓ Total downloaded: {total_size:.1f} MB")
        print(f"✓ Cache location: {cache_dir}")
        print(f"✓ Quantization: {quantization}")
        
        # Verify files exist
        print(f"\n4. Verifying files...")
        onnx_dir = cache_dir / "onnx"
        if not onnx_dir.exists():
            print("✗ ONNX directory not found")
            return False
            
        expected_files = [config[key] for key in config.keys()]
        missing_files = []
        
        for file in expected_files:
            if not (onnx_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
            return False
        else:
            print("✓ All ONNX model files verified")
        
        print(f"\n{'='*60}")
        print("ONNX MODEL DOWNLOAD COMPLETE")
        print(f"{'='*60}")
        print("Models are ready for inference!")
        print("Next step: python3 onnx_app.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download ONNX Gemma 3n E2B models")
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory")
    parser.add_argument("--quantization", choices=["mixed", "max_savings", "high_quality"], 
                       default="mixed", help="Quantization strategy")
    
    args = parser.parse_args()
    
    print("ONNX GEMMA 3N E2B DOWNLOAD")
    print("="*60)
    
    success = download_onnx_models(args.cache_dir, args.quantization)
    
    if not success:
        print("Download failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    print("\nReady to run: python3 onnx_app.py")

if __name__ == "__main__":
    main()
