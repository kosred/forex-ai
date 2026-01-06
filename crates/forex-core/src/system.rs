use crate::config::Settings;
use serde::{Deserialize, Serialize};
use std::env;
use std::process::Command;
use sysinfo::System;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub gpu_names: Vec<String>,
    pub num_gpus: usize,
    pub gpu_mem_gb: Vec<f64>,
    pub timestamp: String,
    pub platform_label: String,
}

pub struct HardwareProbe {
    sys: System,
}

impl HardwareProbe {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        Self { sys }
    }

    pub fn detect(&mut self) -> HardwareProfile {
        self.sys.refresh_all();

        let cpu_cores = self.sys.physical_core_count().unwrap_or(1);
        let total_ram_gb = self.sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
        let available_ram_gb = self.sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

        let (gpu_names, gpu_mem_gb) = self.detect_gpus_nvidia_smi();
        let num_gpus = gpu_names.len();

        let platform_label = format!(
            "{} {}",
            System::name().unwrap_or_default(),
            System::os_version().unwrap_or_default()
        );

        HardwareProfile {
            cpu_cores,
            total_ram_gb,
            available_ram_gb,
            gpu_names,
            num_gpus,
            gpu_mem_gb,
            timestamp: chrono::Utc::now().to_rfc3339(),
            platform_label,
        }
    }

    fn detect_gpus_nvidia_smi(&self) -> (Vec<String>, Vec<f64>) {
        let mut names = Vec::new();
        let mut mems = Vec::new();

        let smi_candidates = if cfg!(target_os = "windows") {
            vec![
                "nvidia-smi",
                r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                r"C:\Windows\System32\nvidia-smi.exe",
            ]
        } else {
            vec!["nvidia-smi"]
        };

        for cmd in smi_candidates {
            if let Ok(output) = Command::new(cmd)
                .args(&["--query-gpu=name", "--format=csv,noheader"])
                .output()
            {
                if output.status.success() {
                    let out_str = String::from_utf8_lossy(&output.stdout);
                    for line in out_str.lines() {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() {
                            names.push(trimmed.to_string());
                        }
                    }
                    if !names.is_empty() {
                        if let Ok(mem_out) = Command::new(cmd)
                            .args(&["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
                            .output()
                        {
                            let mem_str = String::from_utf8_lossy(&mem_out.stdout);
                            for line in mem_str.lines() {
                                if let Ok(mb) = line.trim().parse::<f64>() {
                                    mems.push(mb / 1024.0);
                                }
                            }
                        }
                        return (names, mems);
                    }
                }
            }
        }

        (vec![], vec![])
    }
}

pub struct AutoTuner<'a> {
    settings: &'a mut Settings,
    profile: HardwareProfile,
}

#[derive(Debug, Clone)]
pub struct AutoTuneHints {
    pub enable_gpu: bool,
    pub num_gpus: usize,
    pub device: String,
    pub n_jobs: usize,
    pub train_batch_size: usize,
    pub inference_batch_size: usize,
    pub hpo_trials: usize,
    pub adaptive_training_budget: f64,
    pub feature_workers: usize,
    pub is_hpc: bool,
}

impl<'a> AutoTuner<'a> {
    pub fn new(settings: &'a mut Settings, profile: HardwareProfile) -> Self {
        Self { settings, profile }
    }

    pub fn apply(&mut self) -> AutoTuneHints {
        let hints = self.evaluate_hints();

        // Correct casting based on previous errors: expected usize, found i64 -> so target is usize.

        self.settings.system.enable_gpu = hints.enable_gpu;
        self.settings.system.num_gpus = hints.num_gpus; // usize
        self.settings.system.device = hints.device.clone();
        self.settings.system.n_jobs = hints.n_jobs; // usize

        self.settings.models.train_batch_size = hints.train_batch_size; // usize
        self.settings.models.inference_batch_size = hints.inference_batch_size; // usize
        self.settings.models.hpo_trials = hints.hpo_trials; // usize

        // Apply env vars for threads
        self.apply_thread_env_defaults(&hints);

        info!(
            "Auto-Tuner Applied: GPU={} Device={}",
            hints.enable_gpu, hints.device
        );

        hints
    }

    fn evaluate_hints(&self) -> AutoTuneHints {
        // Fix: enable_gpu_preference is String, so use as_str() instead of as_deref()
        // If it was Option<String>, we'd handl it, but error said String.
        // Wait, if it's String, does it have a default? config.rs suggested defaults.
        // Let's check if it's Option in the struct. config.rs usually has defaults.
        // Assuming it's Option<String> based on "serde(default)", but check error again:
        // "no method named `as_deref` found for struct `std::string::String`" -> It IS String.
        let preference = self.settings.system.enable_gpu_preference.as_str();
        let preference = if preference.is_empty() {
            "auto"
        } else {
            preference
        };

        let has_gpu = self.profile.num_gpus > 0;

        let enable_gpu = match preference {
            "cpu" | "false" | "0" | "no" | "off" => false,
            "gpu" | "cuda" | "true" | "1" | "yes" | "on" => has_gpu,
            _ => has_gpu,
        };

        if preference == "gpu" && !has_gpu {
            warn!("GPU requested but no GPU detected; defaulting to CPU");
        }

        let device = if enable_gpu {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        };

        let cpu_cores = self.profile.cpu_cores.max(1);
        let ram_gb = self.profile.available_ram_gb;
        let min_vram_gb = self
            .profile
            .gpu_mem_gb
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let min_vram_gb = if min_vram_gb == f64::INFINITY {
            0.0
        } else {
            min_vram_gb
        };

        let cpu_budget = self.resolve_cpu_budget(cpu_cores);

        // Simple logic for batch sizes
        let (train_batch, infer_batch) = if enable_gpu {
            if min_vram_gb >= 20.0 {
                (1024, 4096)
            } else if min_vram_gb >= 12.0 {
                (512, 2048)
            } else {
                (256, 1024)
            }
        } else {
            (64, 128)
        };

        // Feature workers logic
        let per_worker_gb = 2.0;
        let ram_based_workers = (ram_gb / per_worker_gb) as usize;
        let feature_workers = 1.max(cpu_budget.min(ram_based_workers));

        AutoTuneHints {
            enable_gpu,
            num_gpus: if enable_gpu { self.profile.num_gpus } else { 0 },
            device,
            n_jobs: cpu_budget,
            train_batch_size: train_batch,
            inference_batch_size: infer_batch,
            hpo_trials: if enable_gpu { 50 } else { 20 },
            adaptive_training_budget: if enable_gpu { 3600.0 } else { 1800.0 },
            feature_workers,
            is_hpc: ram_gb > 64.0 && cpu_cores >= 32,
        }
    }

    fn resolve_cpu_budget(&self, total_cores: usize) -> usize {
        if let Ok(val) = env::var("FOREX_BOT_CPU_BUDGET") {
            if let Ok(n) = val.parse::<usize>() {
                return n.min(total_cores).max(1);
            }
        }
        let reserve = 1;
        if total_cores > reserve {
            total_cores - reserve
        } else {
            1
        }
    }

    fn apply_thread_env_defaults(&self, hints: &AutoTuneHints) {
        let n_threads = hints.n_jobs.max(1).to_string();
        unsafe {
            env::set_var("OMP_NUM_THREADS", &n_threads);
            env::set_var("MKL_NUM_THREADS", &n_threads);
            env::set_var("OPENBLAS_NUM_THREADS", &n_threads);
        }
    }
}
