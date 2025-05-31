use polars::prelude::*;
use std::path::PathBuf;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Dataset {
    pub name: String,
    pub path: PathBuf,
    #[serde(skip)]
    pub df: Option<DataFrame>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub name: String,
    pub datasets: Vec<Dataset>,
    pub selected_dataset: Option<usize>,
    pub subprojects: Vec<Project>,
}

pub struct AppState {
    pub projects: Vec<Project>,
    pub selected_project_path: Vec<usize>,
    pub selected_tab: Tab,
    pub pending_delete: Option<Vec<usize>>,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Tab {
    Preview,
    Plot,
    QC,
}
