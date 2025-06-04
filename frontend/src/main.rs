// use eframe::{egui, App, Frame, NativeOptions, run_native};
// use reqwest::blocking::Client;
// use serde::Deserialize;
// use std::sync::{Arc, Mutex};
// use std::thread;

// #[derive(Deserialize)]
// struct Message {
//     message: String,
// }

// struct AppState {
//     last_message: String,
//     is_fetching: bool,
// }

// struct MyApp {
//     client: Client,
//     state: Arc<Mutex<AppState>>,
// }

// impl Default for MyApp {
//     fn default() -> Self {
//         Self {
//             client: Client::new(),
//             state: Arc::new(Mutex::new(AppState {
//                 last_message: "Press the button…".into(),
//                 is_fetching: false,
//             })),
//         }
//     }
// }

// impl App for MyApp {
//     fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
//         egui::CentralPanel::default().show(ctx, |ui| {
//             let mut st = self.state.lock().unwrap();

//             if st.is_fetching {
//                 ui.label("Fetching…");
//             } else if ui.button("Fetch from Python").clicked() {
//                 st.is_fetching = true;
//                 let client = self.client.clone();
//                 let state = Arc::clone(&self.state);
//                 thread::spawn(move || {
//                     let resp = client
//                         .get("http://127.0.0.1:8000/hello")
//                         .send()
//                         .and_then(|r| r.json::<Message>());
//                     let mut st = state.lock().unwrap();
//                     match resp {
//                         Ok(msg) => st.last_message = msg.message,
//                         Err(err) => st.last_message = format!("Error: {}", err),
//                     }
//                     st.is_fetching = false;
//                 });
//             }

//             ui.separator();
//             ui.label(&st.last_message);
//         });

//         // continuously repaint to catch state updates from the background thread
//         ctx.request_repaint();
//     }
// }

// fn main() {
//     let native_options = NativeOptions::default();
//     run_native(
//         "egui ↔ FastAPI Demo",
//         native_options,
//         Box::new(|_cc| Box::new(MyApp::default())),
//     );
// }

// main.rs

// src/main.rs

extern crate core;

mod models;
mod ui;
mod requests;

use crate::ui::get_project_mut;
use eframe::egui::Vec2;
use eframe::{Error, egui};
use egui::Visuals;
use models::{AppState, Tab};

pub struct MyApp {
    state: AppState,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            state: AppState {
                projects: vec![],
                selected_project_path: vec![],
                selected_tab: Tab::Preview,
                pending_delete: None,
                note_editing: None,
                column_to_remove: None,
                column_to_match: None,
                match_value: "".to_string(),
                graph_pan: Vec2::ZERO,
                graph_zoom: 1.0,
                node_positions: Vec::new(),
                layout_done: false,
                bar_genes: Vec::new(),
                bar_counts: Vec::new(),
                is_fetching: false,
                bar_selected_field: String::new(),
                bar_selected_value: String::new(),
            },
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(Visuals::dark());

        egui::SidePanel::left("project_panel").show(ctx, |ui| {
            ui.set_width(200.0);
            ui.heading("Bio Research Tool");
            ui.separator();
            ui::side_panel(ui, &mut self.state);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui::tab_bar(ui, &mut self.state);
            ui.separator();
            match self.state.selected_tab {
                Tab::Preview => ui::preview_tab(ctx, ui, &mut self.state),
                Tab::Graph => ui::graph_tab(ctx, ui, &mut self.state),
                Tab::BarChart => ui::bar_chart_tab(ctx, ui, &mut self.state),
            }
        });

        if let Some((path, note_idx)) = &mut self.state.note_editing {
            if let Some(proj) = get_project_mut(&mut self.state.projects, path) {
                if let Some(note) = proj.notes.get_mut(*note_idx) {
                    let window_id = egui::Id::new((path.clone(), *note_idx));
                    egui::Window::new(&note.name)
                        .id(window_id)
                        .resizable(true)
                        .collapsible(false)
                        .default_size([400.0, 300.0])
                        .show(ctx, |ui| {
                            ui.horizontal(|ui| {
                                ui.label("Note Name:");
                                ui.text_edit_singleline(&mut note.name);
                            });

                            ui.separator();

                            egui::ScrollArea::vertical().show(ui, |ui| {
                                ui.add(
                                    egui::TextEdit::multiline(&mut note.content)
                                        .desired_rows(20)
                                        .desired_width(f32::INFINITY)
                                        .code_editor(),
                                );
                            });

                            ui.separator();
                            if ui.button("Close").clicked() {
                                self.state.note_editing = None;
                            }
                        });
                }
            }
        }
    }
}

fn main() -> Result<(), Error> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Bio Research Tool",
        options,
        Box::new(|_cc| Ok(Box::new(MyApp::default()))),
    )?;

    Ok(())
}
