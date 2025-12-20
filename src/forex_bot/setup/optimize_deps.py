"""
Auto-detect CPU vendor and install optimized dependencies.

This script:
1. Detects CPU vendor (Intel vs AMD)
2. Checks current NumPy BLAS backend
3. Optionally reinstalls NumPy with optimal BLAS library
4. Verifies installation

Usage:
    python -m forex_bot.setup.optimize_deps --auto-install
    python -m forex_bot.setup.optimize_deps --check-only
"""

import logging
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)


def detect_cpu_vendor() -> str:
    """
    Detect CPU vendor (Intel, AMD, or Unknown).

    Returns:
        'intel', 'amd', or 'unknown'
    """
    try:
        import cpuinfo  # noqa: F401

        info = cpuinfo.get_cpu_info()
        vendor = info.get("vendor_id_raw", "").lower()

        if "intel" in vendor or "genuineintel" in vendor:
            return "intel"
        elif "amd" in vendor or "authenticamd" in vendor:
            return "amd"

    except ImportError:
        logger.warning("py-cpuinfo not installed, trying platform module...")

    # Fallback: platform.processor()
    try:
        proc = platform.processor().lower()
        if "intel" in proc:
            return "intel"
        elif "amd" in proc or "ryzen" in proc or "epyc" in proc:
            return "amd"
    except Exception as e:
        logger.warning(f"CPU detection failed: {e}")

    return "unknown"


def get_numpy_blas_backend() -> dict:
    """
    Detect current NumPy BLAS backend.

    Returns:
        dict with 'backend' (mkl/openblas/blis/unknown) and 'config' (raw output)
    """
    try:
        import numpy as np

        # NumPy 2.0+ uses show_config()
        config = np.show_config()
        config_str = str(config).lower()

        backend = "unknown"
        if "mkl" in config_str or "intel-mkl" in config_str:
            backend = "mkl"
        elif "openblas" in config_str or "scipy-openblas" in config_str:
            backend = "openblas"
        elif "blis" in config_str:
            backend = "blis"
        elif "accelerate" in config_str:
            backend = "accelerate"

        return {"backend": backend, "config": config_str}

    except Exception as e:
        logger.error(f"Failed to detect NumPy BLAS: {e}")
        return {"backend": "unknown", "config": str(e)}


def check_optimal_setup() -> dict:
    """
    Check if current setup is optimal for detected CPU.

    Returns:
        dict with 'cpu_vendor', 'current_blas', 'recommended_blas', 'is_optimal'
    """
    cpu_vendor = detect_cpu_vendor()
    blas_info = get_numpy_blas_backend()
    current_blas = blas_info["backend"]

    # Determine recommended BLAS
    if cpu_vendor == "intel":
        recommended_blas = "mkl"
    elif cpu_vendor == "amd":
        recommended_blas = "openblas"
    else:
        recommended_blas = "openblas"  # Safe default

    is_optimal = current_blas == recommended_blas

    return {
        "cpu_vendor": cpu_vendor,
        "current_blas": current_blas,
        "recommended_blas": recommended_blas,
        "is_optimal": is_optimal,
        "blas_config": blas_info["config"],
    }


def install_optimal_numpy(cpu_vendor: str, force: bool = False) -> bool:
    """
    Install NumPy with optimal BLAS backend for detected CPU.

    Args:
        cpu_vendor: 'intel', 'amd', or 'unknown'
        force: If True, reinstall even if already optimal

    Returns:
        True if installation succeeded
    """
    try:
        # Check current status first
        status = check_optimal_setup()

        if status["is_optimal"] and not force:
            print(f"✓ NumPy already optimized for {cpu_vendor.upper()} CPU ({status['current_blas']})")
            return True

        print(f"\n🔧 Installing optimized NumPy for {cpu_vendor.upper()} CPU...")

        # Uninstall current NumPy
        print("  → Uninstalling current NumPy...")
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy"])

        # Install based on CPU vendor
        if cpu_vendor == "intel":
            print("  → Installing NumPy with Intel MKL (optimized for Intel CPUs)...")
            # Try conda-style first (if available), else pip
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "numpy>=2.0.0",
                        "--index-url",
                        "https://pypi.anaconda.org/intel/simple",
                    ]
                )
            except subprocess.CalledProcessError:
                # Fallback: standard pip with mkl
                print("    → Anaconda MKL failed, trying standard install...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=2.0.0"])

        elif cpu_vendor == "amd":
            print("  → Installing NumPy with OpenBLAS (optimized for AMD CPUs)...")
            # Install scipy-openblas explicitly
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy-openblas32>=0.3.23"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=2.0.0"])

        else:
            print("  → CPU vendor unknown, installing default NumPy with OpenBLAS...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy-openblas32>=0.3.23"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy>=2.0.0"])

        # Verify installation
        print("\n  → Verifying installation...")
        new_status = check_optimal_setup()

        print("\n✅ NumPy installed successfully!")
        print(f"   CPU Vendor: {new_status['cpu_vendor'].upper()}")
        print(f"   BLAS Backend: {new_status['current_blas'].upper()}")
        print(f"   Optimal: {'YES ✓' if new_status['is_optimal'] else 'NO ✗'}")

        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Installation failed: {e}")
        print(f"\n❌ Installation failed: {e}")
        print("   Try manual installation:")
        if cpu_vendor == "intel":
            print("   pip install numpy")
        else:
            print("   pip install scipy-openblas32 numpy")
        return False

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return False


def install_cpuinfo_if_needed():
    """Install py-cpuinfo for better CPU detection."""
    try:
        import cpuinfo  # noqa: F401
    except ImportError:
        print("📦 Installing py-cpuinfo for accurate CPU detection...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "py-cpuinfo>=9.0.0"])
            print("✓ py-cpuinfo installed")
        except subprocess.CalledProcessError:
            print("⚠ Failed to install py-cpuinfo, using fallback detection")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize NumPy BLAS for your CPU")
    parser.add_argument("--check-only", action="store_true", help="Only check current setup, do not install")
    parser.add_argument("--auto-install", action="store_true", help="Automatically install optimal NumPy")
    parser.add_argument("--force", action="store_true", help="Force reinstall even if already optimal")

    args = parser.parse_args()

    # Install cpuinfo for better detection
    install_cpuinfo_if_needed()

    # Check current setup
    print("\n" + "=" * 60)
    print("NumPy BLAS Optimization Check")
    print("=" * 60)

    status = check_optimal_setup()

    print("\n📊 Current Configuration:")
    print(f"   CPU Vendor: {status['cpu_vendor'].upper()}")
    print(f"   Current BLAS: {status['current_blas'].upper()}")
    print(f"   Recommended BLAS: {status['recommended_blas'].upper()}")
    print(f"   Optimal: {'YES ✓' if status['is_optimal'] else 'NO ✗'}")

    if not status["is_optimal"]:
        print("\n💡 Recommendation:")
        print(
            f"   Your {status['cpu_vendor'].upper()} CPU would benefit from "
            f"{status['recommended_blas'].upper()} backend"
        )
        print("   Expected speedup: 2-10x for matrix operations")

    # Auto-install if requested
    if args.auto_install and not (status["is_optimal"] and not args.force):
        print("\n" + "=" * 60)
        response = input("\n⚠️  This will reinstall NumPy. Continue? [y/N]: ")
        if response.lower() in ["y", "yes"]:
            install_optimal_numpy(status["cpu_vendor"], force=args.force)
        else:
            print("Installation cancelled.")

    elif not args.check_only and not status["is_optimal"]:
        print("\n💡 To install optimized NumPy, run:")
        print("   python -m forex_bot.setup.optimize_deps --auto-install")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
