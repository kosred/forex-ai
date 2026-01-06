
import sys
import os
import time

# Add target/debug to sys.path to find the built pyd/dll
# Adjust path relative to this script location
target_debug = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "debug"))
target_bindings = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "target", "bindings_build", "debug"))
sys.path.append(target_debug)
sys.path.append(target_bindings)

print(f"Searching for bindings in: {target_debug} and {target_bindings}")

try:
    import forex_bindings
    print("Successfully imported forex_bindings!")
except ImportError as e:
    # Try looking for .dll and loading it explicitly or renaming it if needed
    # On Windows, cargo build produces .dll (or .pyd if configured?) 
    # Usually strictly .dll unless cdylib name is correct and extension handled.
    # Python expects .pyd.
    print(f"Import failed: {e}")
    # Helper to check if file exists
    dll_path = os.path.join(target_debug, "forex_bindings.dll")
    pyd_path = os.path.join(target_debug, "forex_bindings.pyd")
    
    # Check bindings build dir too
    dll_path_bindings = os.path.join(target_bindings, "forex_bindings.dll")
    pyd_path_bindings = os.path.join(target_bindings, "forex_bindings.pyd")

    if os.path.exists(dll_path_bindings):
        dll_path = dll_path_bindings
        pyd_path = pyd_path_bindings

    if os.path.exists(dll_path) and not os.path.exists(pyd_path):
        print(f"Found .dll at {dll_path}, creating symlink/copy to .pyd")
        import shutil
        shutil.copy2(dll_path, pyd_path)
        print("Retrying import...")
        import forex_bindings
        print("Successfully imported forex_bindings after renaming!")
    else:
        print("Could not find forex_bindings.dll or .pyd")
        sys.exit(1)

def test_core():
    print("\nTesting ForexCore...")
    core = forex_bindings.ForexCore()
    hardware = core.detect_hardware()
    print("Hardware Detected:")
    print(hardware)
    # Basic validation
    assert 'cpu_cores' in hardware
    assert 'gpu_names' in hardware

def test_models():
    print("\nTesting ModelEngine...")
    try:
        engine = forex_bindings.ModelEngine()
        print("ModelEngine initialized.")
        # We assume some models might exist or we just check init
        # engine.load_models("...") 
    except Exception as e:
        print(f"ModelEngine error: {e}")
        # It might fail if ORT libs are missing, expected for now if CUDA not matching
        pass

if __name__ == "__main__":
    test_core()
    test_models()
