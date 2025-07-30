"""
ONNX Gemma 3n E2B Download Script
Downloads specific quantized ONNX models and required files for local inference
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import argparse


def download_onnx_models(cache_dir=None):
    """Download ONNX Gemma 3n E2B models with specific quantization strategy"""
    
    # Repository info
    model_id = "onnx-community/gemma-3n-E2B-it-ONNX"
    
    # Set up cache directory
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "onnx_models" / "gemma-3n-e2b"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("ONNX GEMMA 3N E2B MODEL DOWNLOADER")
    print("="*70)
    print(f"📁 Download location: {cache_dir}")
    print(f"🔗 Repository: {model_id}")
    print("📋 Quantization Strategy:")
    print("   • embed_tokens    → quantized (2.7 GB)")
    print("   • audio_encoder   → fp16 (1.4 GB)")  
    print("   • vision_encoder  → quantized (0.3 GB)")
    print("   • decoder         → q4f16 (1.4 GB)")
    print("   • Total Size      → ~5.8 GB")
    print("-" * 70)
    
    # Define the specific files we need based on your requirements
    files_to_download = {
        # Configuration and tokenizer files (required for processor)
        "config_files": [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json", 
            "special_tokens_map.json",
            "preprocessor_config.json",
            "generation_config.json"
        ],
        
        # ONNX model files with your specific quantization choices
        "onnx_models": {
            "onnx/embed_tokens_quantized.onnx": "embed_tokens (quantized)",
            "onnx/audio_encoder.onnx": "audio_encoder (fp16)", 
            "onnx/vision_encoder_quantized.onnx": "vision_encoder (quantized)",
            "onnx/decoder_model_merged_q4.onnx": "decoder (q4f16)"
        }
    }
    
    total_downloaded = 0
    failed_downloads = []
    
    try:
        # Step 1: Download configuration files
        print("🔧 Step 1: Downloading configuration files...")
        
        for file in files_to_download["config_files"]:
            try:
                print(f"   📥 {file}...", end=" ")
                
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                
                size_kb = os.path.getsize(downloaded_path) / 1024
                print(f"✅ ({size_kb:.1f} KB)")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)[:50]}...")
                failed_downloads.append(file)
        
        print()
        
        # Step 2: Download ONNX model files
        print("🤖 Step 2: Downloading ONNX model files...")
        
        for file_path, description in files_to_download["onnx_models"].items():
            try:
                print(f"   📥 {description}...", end=" ")
                
                downloaded_path = hf_hub_download(
                    repo_id=model_id,
                    filename=file_path,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                
                size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
                total_downloaded += size_mb
                print(f"✅ ({size_mb:.1f} MB)")
                
            except Exception as e:
                print(f"❌ Failed: {str(e)[:50]}...")
                failed_downloads.append(file_path)
        
        print()
        
        # Step 3: Verify downloaded files
        print("🔍 Step 3: Verifying downloaded files...")
        
        onnx_dir = cache_dir / "onnx"
        expected_onnx_files = [
            "embed_tokens_quantized.onnx",
            "audio_encoder.onnx", 
            "vision_encoder_quantized.onnx",
            "decoder_model_merged_q4.onnx"
        ]
        
        all_files_present = True
        for file in expected_onnx_files:
            file_path = onnx_dir / file
            if file_path.exists():
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   ✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {file} - MISSING")
                all_files_present = False
        
        # Check config files
        for file in files_to_download["config_files"]:
            file_path = cache_dir / file
            if file_path.exists():
                print(f"   ✅ {file}")
            else:
                print(f"   ❌ {file} - MISSING")
                all_files_present = False
        
        print()
        
        # Final summary
        if all_files_present and not failed_downloads:
            print("="*70)
            print("🎉 DOWNLOAD COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"📊 Total size downloaded: {total_downloaded:.1f} MB")
            print(f"📁 Files location: {cache_dir}")
            print(f"🚀 Ready to run: python3 app.py")
            print("="*70)
            return True
        else:
            print("="*70)
            print("⚠️  DOWNLOAD COMPLETED WITH ISSUES")
            print("="*70)
            if failed_downloads:
                print("❌ Failed downloads:")
                for file in failed_downloads:
                    print(f"   • {file}")
            if not all_files_present:
                print("❌ Some files are missing from verification")
            print("🔄 Try running the script again")
            return False
            
    except Exception as e:
        print(f"\n💥 Critical error during download: {e}")
        return False


def check_requirements():
    """Check if required packages are installed"""
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("❌ Missing required package: huggingface_hub")
        print("📦 Install with: pip install huggingface_hub")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX Gemma 3n E2B models with specific quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 download.py                           # Download to default cache
  python3 download.py --cache-dir ./models     # Download to custom directory
  python3 download.py --check                  # Just check what's already downloaded
        """
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        help="Custom cache directory (default: ~/.cache/onnx_models/gemma-3n-e2b)"
    )
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check what files are already present without downloading"
    )
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Set up cache directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = Path.home() / ".cache" / "onnx_models" / "gemma-3n-e2b"
    
    # Check mode - just verify what's present
    if args.check:
        print("🔍 CHECKING EXISTING FILES")
        print("="*50)
        
        if not cache_dir.exists():
            print(f"❌ Cache directory doesn't exist: {cache_dir}")
            sys.exit(1)
        
        # Check ONNX files
        onnx_dir = cache_dir / "onnx"
        expected_files = [
            "embed_tokens_quantized.onnx",
            "audio_encoder.onnx",
            "vision_encoder_quantized.onnx", 
            "decoder_model_merged_q4.onnx"
        ]
        
        all_present = True
        total_size = 0
        
        for file in expected_files:
            file_path = onnx_dir / file
            if file_path.exists():
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                print(f"✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"❌ {file} - MISSING")
                all_present = False
        
        if all_present:
            print(f"\n🎉 All files present! Total: {total_size:.1f} MB")
            print("🚀 Ready to run: python3 app.py")
        else:
            print(f"\n⚠️  Some files missing. Run without --check to download.")
        
        sys.exit(0)
    
    # Normal download mode
    success = download_onnx_models(args.cache_dir)
    
    if not success:
        print("\n💥 Download failed. Please check:")
        print("   • Internet connection")
        print("   • Disk space (~6 GB needed)")
        print("   • Try running again (resume from partial download)")
        sys.exit(1)


if __name__ == "__main__":
    main()
