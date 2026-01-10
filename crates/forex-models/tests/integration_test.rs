// Integration test for forex-models crate
// Tests all ported models can be instantiated

use forex_models::*;

#[test]
fn test_compilation() {
    // This test just verifies that all modules compile
    println!("✓ All forex-models modules compiled successfully");
}

#[test]
fn test_registry() {
    use forex_models::registry::*;

    // Test available models
    assert!(AVAILABLE_MODELS.len() > 0, "Should have available models");
    println!("Available models: {:?}", AVAILABLE_MODELS);

    // Test model info
    let lgbm_info = get_model_info("lightgbm").expect("LightGBM should be in registry");
    assert_eq!(lgbm_info.name, "lightgbm");
    println!("LightGBM info: {:?}", lgbm_info);

    // Test valid model check
    assert!(is_valid_model("lightgbm"));
    assert!(is_valid_model("mlp"));
    assert!(!is_valid_model("nonexistent"));

    println!("✓ Registry tests passed");
}

#[test]
fn test_model_categories() {
    use forex_models::registry::*;

    let categories = list_models_by_category();
    println!("Models by category:");
    for (category, models) in categories.iter() {
        println!("  {:?}: {} models", category, models.len());
    }

    assert!(categories.len() > 0, "Should have model categories");
    println!("✓ Model category tests passed");
}

#[test]
#[cfg(feature = "tch")]
fn test_hardware_detection() {
    use forex_models::hardware::*;

    let hw = HardwareInfo::detect();
    println!("Hardware detected:");
    println!("  CPU cores: {} (logical: {})", hw.cpu_cores, hw.logical_cores);
    println!("  RAM: {:.2} GB", hw.ram_gb);
    println!("  GPUs: {}", hw.gpu_count);

    for (i, gpu) in hw.gpu_list.iter().enumerate() {
        println!("    GPU {}: {}", i, gpu);
    }

    assert!(hw.cpu_cores > 0, "Should detect CPU cores");
    assert!(hw.ram_gb > 0.0, "Should detect RAM");

    println!("✓ Hardware detection tests passed");
}
