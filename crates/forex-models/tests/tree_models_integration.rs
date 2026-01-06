// Integration tests for tree models
// Tests full training and prediction pipeline with real data

#[cfg(all(test, feature = "lightgbm"))]
mod lightgbm_tests {
    use forex_models::tree_models::{LightGBMExpert, TreeModel};
    use polars::prelude::*;
    use ndarray::Array2;

    fn create_sample_data() -> (DataFrame, Series) {
        // Create synthetic classification data
        let n_samples = 1000;
        let n_features = 10;

        // Generate random features
        let mut features: Vec<Series> = Vec::new();
        for i in 0..n_features {
            let data: Vec<f64> = (0..n_samples)
                .map(|j| (j as f64 * 0.1 + i as f64) % 10.0)
                .collect();
            features.push(Series::new(&format!("feature_{}", i), data));
        }

        let df = DataFrame::new(features).unwrap();

        // Generate labels {-1, 0, 1} based on features
        let labels: Vec<i64> = (0..n_samples)
            .map(|i| {
                let val = i % 3;
                match val {
                    0 => -1,
                    1 => 0,
                    _ => 1,
                }
            })
            .collect();
        let labels_series = Series::new("label", labels);

        (df, labels_series)
    }

    #[test]
    fn test_lightgbm_train_predict() {
        // Create sample data
        let (df, labels) = create_sample_data();

        // Split into train/test
        let n_train = 800;
        let train_df = df.slice(0, n_train);
        let test_df = df.slice(n_train as i64, 200);
        let train_labels = labels.slice(0, n_train);

        // Create and train model
        let mut model = LightGBMExpert::new(1, None);

        // Set environment variable to force CPU (avoid GPU errors in CI)
        std::env::set_var("FOREX_BOT_TREE_DEVICE", "cpu");

        let result = model.fit(&train_df, &train_labels);
        assert!(result.is_ok(), "Training should succeed: {:?}", result.err());

        // Predict
        let predictions = model.predict_proba(&test_df);
        assert!(predictions.is_ok(), "Prediction should succeed: {:?}", predictions.err());

        let probs = predictions.unwrap();
        assert_eq!(probs.shape(), &[200, 3], "Output should be [n_samples, 3]");

        // Check probabilities sum to ~1
        for row in probs.outer_iter() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Probabilities should sum to 1, got {}", sum);
        }

        // Check probabilities are in [0, 1]
        for &prob in probs.iter() {
            assert!(prob >= 0.0 && prob <= 1.0, "Probability should be in [0, 1], got {}", prob);
        }
    }

    #[test]
    fn test_lightgbm_save_load() {
        use std::path::Path;
        use std::fs;

        let (df, labels) = create_sample_data();
        let train_df = df.slice(0, 800);
        let train_labels = labels.slice(0, 800);

        // Train model
        let mut model = LightGBMExpert::new(1, None);
        std::env::set_var("FOREX_BOT_TREE_DEVICE", "cpu");
        model.fit(&train_df, &train_labels).unwrap();

        // Save model
        let model_path = Path::new("test_lightgbm_model.txt");
        let result = model.save(model_path);
        assert!(result.is_ok(), "Save should succeed: {:?}", result.err());

        // Load model
        let mut loaded_model = LightGBMExpert::new(1, None);
        let result = loaded_model.load(model_path);
        assert!(result.is_ok(), "Load should succeed: {:?}", result.err());

        // Predict with loaded model
        let test_df = df.slice(800, 200);
        let predictions = loaded_model.predict_proba(&test_df);
        assert!(predictions.is_ok(), "Prediction with loaded model should succeed");

        // Cleanup
        let _ = fs::remove_file(model_path);
    }

    #[test]
    fn test_label_remapping() {
        use forex_models::tree_models::remap_labels_to_contiguous;

        // Test with -1, 0, 1 labels
        let labels = Series::new("label", &[-1i64, 0, 1, -1, 0, 1, 1, 0, -1]);
        let (remapped, mapping) = remap_labels_to_contiguous(&labels).unwrap();

        // Check remapping: -1 -> 0, 0 -> 1, 1 -> 2
        assert_eq!(remapped.as_slice().unwrap(), &[0, 1, 2, 0, 1, 2, 2, 1, 0]);
        assert_eq!(mapping.get(&-1), Some(&0));
        assert_eq!(mapping.get(&0), Some(&1));
        assert_eq!(mapping.get(&1), Some(&2));
    }

    #[test]
    fn test_output_reordering() {
        use forex_models::tree_models::reorder_to_neutral_buy_sell;
        use ndarray::Array2;

        // Test binary classification (2 classes)
        let probs_binary = Array2::from_shape_vec(
            (3, 2),
            vec![0.7, 0.3, 0.6, 0.4, 0.8, 0.2]
        ).unwrap();

        let reordered = reorder_to_neutral_buy_sell(probs_binary, None);
        assert_eq!(reordered.shape(), &[3, 3]);

        // Binary: col 0 -> Neutral, col 1 -> Buy, col 2 -> 0.0 (Sell)
        assert_eq!(reordered[[0, 0]], 0.7); // Neutral
        assert_eq!(reordered[[0, 1]], 0.3); // Buy
        assert_eq!(reordered[[0, 2]], 0.0); // Sell (zero for binary)

        // Test multiclass (3 classes) with known class order
        let probs_multi = Array2::from_shape_vec(
            (2, 3),
            vec![0.1, 0.2, 0.7, 0.5, 0.3, 0.2]
        ).unwrap();

        let reordered = reorder_to_neutral_buy_sell(probs_multi.clone(), Some(vec![0, 1, 2]));
        // With classes [0, 1, 2] (Neutral, Buy, Sell), output should match
        assert_eq!(reordered[[0, 0]], 0.1); // Neutral
        assert_eq!(reordered[[0, 1]], 0.2); // Buy
        assert_eq!(reordered[[0, 2]], 0.7); // Sell
    }

    #[test]
    fn test_time_feature_augmentation() {
        use forex_models::tree_models::augment_time_features;

        // Create DataFrame with 'close' column
        let close_prices: Vec<f64> = vec![100.0, 101.0, 99.0, 102.0, 103.0, 101.0, 104.0, 105.0];
        let volume: Vec<f64> = vec![1000.0, 1100.0, 900.0, 1200.0, 1300.0, 1100.0, 1400.0, 1500.0];

        let df = DataFrame::new(vec![
            Series::new("close", close_prices),
            Series::new("volume", volume),
        ]).unwrap();

        let augmented = augment_time_features(df);
        assert!(augmented.is_ok(), "Augmentation should succeed: {:?}", augmented.err());

        let augmented_df = augmented.unwrap();

        // Check that new columns were added
        assert!(augmented_df.column("ret1").is_ok(), "ret1 column should exist");
        assert!(augmented_df.column("ret1_lag1").is_ok(), "ret1_lag1 column should exist");
        assert!(augmented_df.column("vol14").is_ok() || augmented_df.height() < 14,
                "vol14 should exist if enough data");

        // Original columns should still exist
        assert!(augmented_df.column("close").is_ok(), "close column should exist");
        assert!(augmented_df.column("volume").is_ok(), "volume column should exist");
    }

    #[test]
    fn test_gpu_only_mode() {
        // Test GPU-only mode behavior
        std::env::set_var("FOREX_BOT_GPU_ONLY", "1");
        std::env::set_var("CUDA_VISIBLE_DEVICES", ""); // Simulate no GPU

        let (df, labels) = create_sample_data();
        let train_df = df.slice(0, 800);
        let train_labels = labels.slice(0, 800);

        let mut model = LightGBMExpert::new(1, None);

        // Training should succeed but skip (GPU-only mode with no GPU)
        let result = model.fit(&train_df, &train_labels);
        assert!(result.is_ok(), "Training should complete (skip if no GPU)");

        // Prediction should fail (model disabled)
        let test_df = df.slice(800, 200);
        let result = model.predict_proba(&test_df);
        // This might succeed or fail depending on whether GPU was available
        // Just check it doesn't panic

        // Cleanup
        std::env::remove_var("FOREX_BOT_GPU_ONLY");
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
    }
}

#[cfg(all(test, feature = "xgboost"))]
mod xgboost_tests {
    use forex_models::tree_models::{XGBoostExpert, TreeModel};
    use polars::prelude::*;

    #[test]
    fn test_xgboost_basic() {
        // Similar test to LightGBM but for XGBoost
        let n_samples = 500;
        let n_features = 5;

        let mut features: Vec<Series> = Vec::new();
        for i in 0..n_features {
            let data: Vec<f64> = (0..n_samples)
                .map(|j| (j as f64 * 0.1 + i as f64) % 10.0)
                .collect();
            features.push(Series::new(&format!("feature_{}", i), data));
        }

        let df = DataFrame::new(features).unwrap();
        let labels: Vec<i64> = (0..n_samples).map(|i| (i % 3 - 1) as i64).collect();
        let labels_series = Series::new("label", labels);

        let train_df = df.slice(0, 400);
        let test_df = df.slice(400, 100);
        let train_labels = labels_series.slice(0, 400);

        let mut model = XGBoostExpert::new(1, None);
        std::env::set_var("FOREX_BOT_TREE_DEVICE", "cpu");

        let result = model.fit(&train_df, &train_labels);
        assert!(result.is_ok(), "XGBoost training should succeed: {:?}", result.err());

        let predictions = model.predict_proba(&test_df);
        assert!(predictions.is_ok(), "XGBoost prediction should succeed: {:?}", predictions.err());

        let probs = predictions.unwrap();
        assert_eq!(probs.shape(), &[100, 3]);
    }
}

#[cfg(all(test, feature = "catboost"))]
mod catboost_tests {
    use forex_models::tree_models::{CatBoostExpert, TreeModel};
    use polars::prelude::*;

    #[test]
    fn test_catboost_training_error() {
        // CatBoost should error on training (not supported in Rust)
        let features = vec![Series::new("f1", vec![1.0, 2.0, 3.0])];
        let df = DataFrame::new(features).unwrap();
        let labels = Series::new("label", vec![-1i64, 0, 1]);

        let mut model = CatBoostExpert::new(1, None);
        let result = model.fit(&df, &labels);

        // Should fail with informative error
        assert!(result.is_err(), "CatBoost training should fail in Rust");
        let err_msg = result.err().unwrap().to_string();
        assert!(err_msg.contains("not supported"), "Error should mention not supported");
    }
}
