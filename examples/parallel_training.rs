// Example: Train all 11 models in PARALLEL using Rayon
// Each model releases GIL during training - true multi-core utilization!

use rayon::prelude::*;
use std::time::Instant;

fn main() {
    println!("=".repeat(80));
    println!("PARALLEL MODEL TRAINING - 11 Models Simultaneously");
    println!("=".repeat(80));

    // All 11 available models
    let model_names = vec![
        "lightgbm",
        "xgboost",
        "xgboost_rf",
        "xgboost_dart",
        "catboost",
        "catboost_alt",
        "mlp",
        "nbeats",
        "tide",
        "tabnet",
        "kan",
    ];

    println!("\n[INFO] Training {} models in parallel...", model_names.len());
    println!("[INFO] Each model will use multiple cores (GIL released during training)");
    println!("[INFO] Rayon will distribute models across available CPU cores\n");

    let start = Instant::now();

    // PARALLEL TRAINING - uses ALL cores!
    let results: Vec<_> = model_names
        .par_iter()  // <-- Rayon parallel iterator
        .map(|model_name| {
            let model_start = Instant::now();

            // This is pseudocode - actual implementation would call Rust wrappers
            println!("[START] Training {} on thread {:?}", model_name, std::thread::current().id());

            // Simulate training (in real code, this would be model.fit())
            // The Python library releases GIL here, allowing true parallelism
            std::thread::sleep(std::time::Duration::from_millis(500));

            let duration = model_start.elapsed();
            println!("[DONE]  {} completed in {:.2}s", model_name, duration.as_secs_f64());

            (model_name.to_string(), duration)
        })
        .collect();

    let total_duration = start.elapsed();

    println!("\n{}", "=".repeat(80));
    println!("TRAINING COMPLETE");
    println!("{}", "=".repeat(80));
    println!("Total time: {:.2}s (parallel)", total_duration.as_secs_f64());
    println!("Sequential time would be: ~5.5s (11 models × 0.5s each)");
    println!("Speedup: {:.1}x", 5.5 / total_duration.as_secs_f64());
    println!("\n[SUCCESS] All {} models trained using parallel execution!", results.len());
    println!("[INFO] Each model used multiple cores internally (GIL released)");
    println!("[INFO] Rayon distributed models across all CPU threads");
}
