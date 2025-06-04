use eframe::egui::Vec2;
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
pub struct Note {
    pub name: String,
    pub content: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Project {
    pub name: String,
    pub datasets: Vec<Dataset>,
    pub selected_dataset: Option<usize>,
    pub subprojects: Vec<Project>,
    pub notes: Vec<Note>,
}

pub struct AppState {
    pub projects: Vec<Project>,
    pub selected_project_path: Vec<usize>,
    pub selected_tab: Tab,
    pub pending_delete: Option<Vec<usize>>,
    pub note_editing: Option<(Vec<usize>, usize)>,
    pub column_to_remove: Option<String>,
    pub column_to_match: Option<String>,
    pub match_value: String,
    pub graph_pan: Vec2,
    pub graph_zoom: f32,
    pub node_positions: Vec<Vec2>,
    pub layout_done: bool,
    pub bar_genes: Vec<String>,
    pub bar_counts: Vec<f64>,
    pub is_fetching: bool,
    pub bar_selected_field: String,
    pub bar_selected_value: String,
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Tab {
    Preview,
    Graph,
    BarChart,
}
