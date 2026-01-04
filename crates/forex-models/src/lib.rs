use std::collections::HashMap;

use ndarray::Array2;

#[derive(Debug, Clone, Copy)]
pub enum Device {
    Cpu,
    Gpu(usize),
}

pub trait Model {
    fn name(&self) -> &str;
    fn fit(&mut self, features: &Array2<f32>, labels: &[f32]);
    fn predict(&self, features: &Array2<f32>) -> Vec<f32>;
}

pub struct NoopModel;

impl Model for NoopModel {
    fn name(&self) -> &str {
        "noop"
    }

    fn fit(&mut self, _features: &Array2<f32>, _labels: &[f32]) {}

    fn predict(&self, features: &Array2<f32>) -> Vec<f32> {
        vec![0.0; features.nrows()]
    }
}

pub type ModelFactory = fn() -> Box<dyn Model + Send + Sync>;

#[derive(Default)]
pub struct ModelRegistry {
    entries: HashMap<String, ModelFactory>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            entries: HashMap::new(),
        };
        registry.register("noop", || Box::new(NoopModel));
        registry
    }

    pub fn register(&mut self, name: &str, factory: ModelFactory) {
        self.entries.insert(name.to_string(), factory);
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn Model + Send + Sync>> {
        self.entries.get(name).map(|f| f())
    }

    pub fn list(&self) -> Vec<String> {
        let mut out: Vec<String> = self.entries.keys().cloned().collect();
        out.sort();
        out
    }
}

#[cfg(feature = "tch")]
pub mod tch_models {
    use super::{Device, Model};
    use ndarray::Array2;
    use tch::{nn, Device as TchDevice, Kind, Tensor};

    pub struct TchMlp {
        vs: nn::VarStore,
        net: nn::Sequential,
        device: TchDevice,
    }

    impl TchMlp {
        pub fn new(input_dim: i64, hidden: i64, output_dim: i64, device: Device) -> Self {
            let device = match device {
                Device::Cpu => TchDevice::Cpu,
                Device::Gpu(idx) => TchDevice::Cuda(idx as i32),
            };
            let vs = nn::VarStore::new(device);
            let root = &vs.root();
            let net = nn::seq()
                .add(nn::linear(root / "l1", input_dim, hidden, Default::default()))
                .add_fn(|x| x.relu())
                .add(nn::linear(root / "l2", hidden, output_dim, Default::default()));
            Self { vs, net, device }
        }
    }

    impl Model for TchMlp {
        fn name(&self) -> &str {
            "tch_mlp"
        }

        fn fit(&mut self, features: &Array2<f32>, labels: &[f32]) {
            let n_rows = features.nrows() as i64;
            let n_cols = features.ncols() as i64;
            if labels.len() != features.nrows() {
                return;
            }
            let x = Tensor::of_slice(features.as_slice().unwrap())
                .view([n_rows, n_cols])
                .to_device(self.device);
            let y = Tensor::of_slice(labels)
                .view([n_rows, 1])
                .to_device(self.device);
            let mut opt = nn::Adam::default().build(&self.vs, 1e-3).ok();
            if let Some(ref mut opt) = opt {
                for _ in 0..5 {
                    let preds = self.net.forward(&x);
                    let loss = preds.mse_loss(&y, tch::Reduction::Mean);
                    opt.backward_step(&loss);
                }
            }
        }

        fn predict(&self, features: &Array2<f32>) -> Vec<f32> {
            let n_rows = features.nrows() as i64;
            let n_cols = features.ncols() as i64;
            let x = Tensor::of_slice(features.as_slice().unwrap())
                .view([n_rows, n_cols])
                .to_device(self.device);
            let preds = self.net.forward(&x);
            Vec::<f32>::from(preds.to_device(TchDevice::Cpu).to_kind(Kind::Float))
        }
    }
}
