import os
import sys

def apply_windows_benchmark_config():
    config_path = os.path.join(os.path.dirname(__file__), 'stress_challenge', 'config.py')
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    with open(config_path, 'r') as f:
        content = f.read()

    # Increase CPU matrix size for Ultra 9 275HX (huge L3 cache)
    content = content.replace("CPU_MATRIX_SIZE = 1024", "CPU_MATRIX_SIZE = 4096")
    
    # Optionally tweak thermal limits for high-end laptops
    content = content.replace("CPU_THROTTLE_TEMP_C = 95", "CPU_THROTTLE_TEMP_C = 100")
    content = content.replace("CPU_CRITICAL_TEMP = 93", "CPU_CRITICAL_TEMP = 98")
    content = content.replace("CPU_WARNING_TEMP = 88", "CPU_WARNING_TEMP = 93")
    content = content.replace("CPU_SAFE_TEMP = 82", "CPU_SAFE_TEMP = 85")
    
    with open(config_path, 'w') as f:
        f.write(content)

    print("✅ High-Performance Windows Config Applied:")
    print("   - CPU_MATRIX_SIZE increased to 4096")
    print("   - CPU limits shifted up by ~5°C for high-end gaming laptops")
    print("   - GPU automatically scales based on 5080 VRAM (up to 85%)")

if __name__ == "__main__":
    apply_windows_benchmark_config()
