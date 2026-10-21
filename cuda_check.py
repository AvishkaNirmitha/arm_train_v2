import torch
import subprocess
import sys

def check_cuda():
    print("🔍 Checking CUDA availability...\n")

    # Check if PyTorch detects CUDA
    if torch.cuda.is_available():
        print("✅ CUDA is available!")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        print(f"PyTorch device count: {torch.cuda.device_count()}\n")

        # Print device info
        for i in range(torch.cuda.device_count()):
            print(f"--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            print(f"Capability: {torch.cuda.get_device_capability(i)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB\n")
    else:
        print("❌ CUDA is NOT available.")
        print("Check if your NVIDIA drivers and CUDA toolkit are properly installed.\n")

def check_nvidia_smi():
    print("🔍 Checking NVIDIA System Management Interface (nvidia-smi)...\n")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi output:\n")
            print(result.stdout)
        else:
            print("⚠️ nvidia-smi returned an error:\n")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ nvidia-smi not found. NVIDIA driver may not be installed or accessible.")

if __name__ == "__main__":
    print(f"Python executable: {sys.executable}")
    print(f"Using torch version: {torch.__version__}\n")

    check_cuda()
    # check_nvidia_smi()