use crate::models::{AppState, Dataset, Project, Tab};
use eframe::egui;
use egui::{Align2, Color32};
use egui::{Direction, Layout, Ui};
use egui_extras::{Column, TableBuilder};
use egui_plot::{Bar, BarChart, Plot};
use polars::datatypes::DataType;
use polars::prelude::{CsvReader, SerReader};
use polars::prelude::{DataFrame, Series};
use rfd::FileDialog;
use std::collections::HashMap;

fn get_project_mut<'a>(projects: &'a mut [Project], path: &[usize]) -> Option<&'a mut Project> {
    if path.is_empty() {
        return None;
    }
    let first = path[0];
    if first >= projects.len() {
        return None;
    }
    let proj = &mut projects[first];
    if path.len() == 1 {
        Some(proj)
    } else {
        get_project_mut(&mut proj.subprojects, &path[1..])
    }
}

fn delete_project_at_path(projects: &mut Vec<Project>, path: &[usize]) {
    if path.is_empty() {
        return;
    }
    if path.len() == 1 {
        let idx = path[0];
        if idx < projects.len() {
            projects.remove(idx);
        }
    } else {
        let parent_path = &path[..path.len() - 1];
        if let Some(parent) = get_project_mut(projects, parent_path) {
            let idx = path[path.len() - 1];
            if idx < parent.subprojects.len() {
                parent.subprojects.remove(idx);
            }
        }
    }
}

fn draw_project_tree(
    ui: &mut Ui,
    proj: &Project,
    path_so_far: &Vec<usize>,
    level: usize,
    state: &AppState,
) -> Option<Vec<usize>> {
    let mut clicked_path: Option<Vec<usize>> = None;

    let is_selected = state.selected_project_path == *path_so_far;

    ui.horizontal(|ui| {
        ui.add_space(level as f32 * 16.0);
        if ui.selectable_label(is_selected, &proj.name).clicked() {
            clicked_path = Some(path_so_far.clone());
        }
    });

    if !proj.datasets.is_empty() {
        for ds in &proj.datasets {
            ui.horizontal(|ui| {
                ui.add_space((level + 1) as f32 * 16.0);
                ui.label(&ds.name);
            });
        }
    }

    for (child_idx, child_proj) in proj.subprojects.iter().enumerate() {
        let mut child_path = path_so_far.clone();
        child_path.push(child_idx);
        if clicked_path.is_none() {
            if let Some(found) = draw_project_tree(ui, child_proj, &child_path, level + 1, state) {
                clicked_path = Some(found);
            }
        }
    }

    clicked_path
}

pub fn side_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Projects");

    let mut maybe_clicked: Option<Vec<usize>> = None;
    for (i, proj) in state.projects.iter().enumerate() {
        let initial_path = vec![i];
        if maybe_clicked.is_none() {
            if let Some(clicked) = draw_project_tree(ui, proj, &initial_path, 0, state) {
                maybe_clicked = Some(clicked);
            }
        }
    }

    if let Some(new_path) = maybe_clicked {
        state.selected_project_path = new_path;
    }

    ui.separator();

    if !state.selected_project_path.is_empty() {
        ui.horizontal(|ui| {
            ui.label("Project name:");
            if let Some(sel_proj) =
                get_project_mut(&mut state.projects, &state.selected_project_path)
            {
                ui.text_edit_singleline(&mut sel_proj.name);
            }
        });
        ui.add_space(8.0);
    }

    if ui.button("＋ Add Root Project").clicked() {
        state.projects.push(Project {
            name: "New".into(),
            subprojects: Vec::new(),
            datasets: Vec::new(),
            selected_dataset: None,
        });
        let idx = state.projects.len() - 1;
        state.selected_project_path = vec![idx];
    }

    if !state.selected_project_path.is_empty() {
        ui.add_space(4.0);
        if ui.button("＋ Add Subproject").clicked() {
            if let Some(parent) = get_project_mut(&mut state.projects, &state.selected_project_path)
            {
                parent.subprojects.push(Project {
                    name: "New".into(),
                    subprojects: Vec::new(),
                    datasets: Vec::new(),
                    selected_dataset: None,
                });
                let child_idx = parent.subprojects.len() - 1;
                let mut new_path = state.selected_project_path.clone();
                new_path.push(child_idx);
                state.selected_project_path = new_path;
            }
        }
    }

    if !state.selected_project_path.is_empty() {
        ui.add_space(8.0);
        if ui.button("🗑 Delete Project").clicked() {
            state.pending_delete = Some(state.selected_project_path.clone());
        }
    }

    ui.separator();

    if let Some(path_to_delete) = state.pending_delete.clone() {
        egui::Window::new("Confirm Deletion")
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(
                        "Are you sure you want to delete this project\nand all of its subprojects?",
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Yes, Delete").clicked() {
                            delete_project_at_path(&mut state.projects, &path_to_delete);
                            state.selected_project_path.clear();
                            state.pending_delete = None;
                        }
                        if ui.button("Cancel").clicked() {
                            state.pending_delete = None;
                        }
                    });
                });
            });
    }
}

pub fn tab_bar(ui: &mut egui::Ui, state: &mut AppState) {
    ui.horizontal(|ui| {
        for &tab in &[Tab::Preview, Tab::Plot, Tab::QC] {
            let label = format!("{:?}", tab);
            if ui
                .selectable_label(state.selected_tab == tab, label)
                .clicked()
            {
                state.selected_tab = tab;
            }
        }
    });
}

pub fn preview_tab(ctx: &egui::Context, ui: &mut egui::Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            ui.vertical(|ui| {
                ui.heading("Datasets");
                for (ds_idx, ds) in proj.datasets.iter().enumerate() {
                    if ui
                        .selectable_label(proj.selected_dataset == Some(ds_idx), &ds.name)
                        .clicked()
                    {
                        proj.selected_dataset = Some(ds_idx);
                    }
                }

                if ui.button("＋ Add CSV").clicked() {
                    if let Some(path) = FileDialog::new().add_filter("CSV", &["csv"]).pick_file() {
                        let name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("csv")
                            .to_string();

                        let result: polars::prelude::PolarsResult<DataFrame> =
                            CsvReader::from_path(&path).and_then(|r| {
                                r.has_header(true)
                                    .infer_schema(Some(10000))
                                    .with_ignore_errors(true)
                                    .finish()
                            });

                        if let Ok(df) = result {
                            proj.datasets.push(Dataset {
                                name: name.clone(),
                                path: path.clone(),
                                df: Some(df),
                            });
                            proj.selected_dataset = Some(proj.datasets.len() - 1);
                            ctx.request_repaint();
                        }
                    }
                }
                ui.label(format!("({} loaded)", proj.datasets.len()));
            });

            ui.separator();

            ui.vertical(|ui| {
                egui::ScrollArea::both().show(ui, |ui| {
                    if let Some(ds_idx) = proj.selected_dataset {
                        if let Some(ds) = proj.datasets.get(ds_idx) {
                            if let Some(df) = &ds.df {
                                let cols = df.get_columns();
                                let rows = df.height();

                                let mut builder = TableBuilder::new(ui).striped(true).cell_layout(
                                    Layout::centered_and_justified(Direction::LeftToRight),
                                );

                                for _ in cols.iter() {
                                    builder = builder.column(Column::auto());
                                }

                                let table = builder.header(20.0, |mut header| {
                                    for series in cols.iter() {
                                        header.col(|ui| {
                                            ui.heading(series.name());
                                        });
                                    }
                                });

                                table.body(|body| {
                                    body.rows(18.0, rows, |mut row| {
                                        let row_idx = row.index();
                                        for series in cols.iter() {
                                            let val = series.get(row_idx);
                                            row.col(|ui| {
                                                ui.label(format!("{:?}", val));
                                            });
                                        }
                                    });
                                });
                            } else {
                                ui.label("Dataset loaded but empty.");
                            }
                        }
                    } else {
                        if proj.datasets.is_empty() {
                            ui.label("No dataset loaded.");
                        } else {
                            ui.label("Select a dataset to preview.");
                        }
                    }
                });
            });
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.label("Select or create a project first.");
        });
    }
}

pub fn plot_tab(ui: &mut egui::Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get(ds_idx) {
                    if let Some(df) = &ds.df {
                        let mut grouped: DataFrame =
                            df.groupby(&["GeneSymbol"]).unwrap().count().unwrap();

                        let first_count_col: String = grouped
                            .get_column_names()
                            .iter()
                            .find(|name| name.ends_with("_count"))
                            .expect("Expected at least one column ending in \"_count\"")
                            .to_string();

                        grouped.rename(&first_count_col, "count").unwrap();

                        let counts_series: Series = grouped
                            .column("count")
                            .unwrap()
                            .cast(&DataType::Int64)
                            .unwrap()
                            .clone();

                        let counts_chunked = counts_series.i64().unwrap();
                        let count_vec: Vec<i64> = counts_chunked.into_no_null_iter().collect();

                        let mut freq_map: HashMap<i64, i64> = HashMap::new();
                        for &num_diseases in &count_vec {
                            *freq_map.entry(num_diseases).or_insert(0) += 1;
                        }

                        let mut bins: Vec<(f64, f64)> = freq_map
                            .into_iter()
                            .map(|(num_diseases, num_genes)| {
                                (num_diseases as f64, num_genes as f64)
                            })
                            .collect();
                        bins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                        let bars: Vec<Bar> = bins
                            .into_iter()
                            .map(|(x, _)| {
                                let bar = Bar::new(x, 0.8);
                                bar
                            })
                            .collect();

                        Plot::new("diseases_per_gene_histogram")
                            .view_aspect(2.0)
                            .show(ui, |plot_ui| {
                                plot_ui.bar_chart(
                                    BarChart::new(bars).color(Color32::from_rgb(100, 150, 250)),
                                );
                            });
                    } else {
                        ui.label("Dataset loaded but empty.");
                    }
                } else {
                    ui.label("Invalid dataset index.");
                }
            } else {
                ui.label("Select a dataset to plot.");
            }
        });
    } else {
        // No project selected at all
        ui.centered_and_justified(|ui| {
            ui.label("Select or create a project first.");
        });
    }
}

pub fn qc_tab(ui: &mut egui::Ui, _state: &mut AppState) {
    ui.centered_and_justified(|ui| {
        ui.label("🔍 QC tab not implemented yet");
    });
}
