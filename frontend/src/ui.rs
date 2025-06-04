use crate::models::{AppState, Dataset, Note, Project, Tab};
use eframe::egui;
use eframe::egui::{Align2, Color32, Pos2, Rect, Sense, Ui, Vec2};
use eframe::emath::RectTransform;
use egui::{Direction, Layout};
use egui_extras::{Column, TableBuilder};
use egui_plot::{Bar, BarChart, Plot};
use polars::datatypes::BooleanChunked;
use polars::error::PolarsResult;
use polars::prelude::DataFrame;
use polars::prelude::{ChunkCompare, DataType};
use polars::prelude::{CsvReader, SerReader};
use rfd::FileDialog;
use std::ops::Not;
use futures::executor::block_on;
use serde_json::Value;
use crate::requests::{request_bar_chart_blocking};

pub fn get_project_mut<'a>(projects: &'a mut [Project], path: &[usize]) -> Option<&'a mut Project> {
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
    ui: &mut egui::Ui,
    proj: &Project,
    path_so_far: &Vec<usize>,
    level: usize,
    state: &AppState,
) -> Option<(Vec<usize>, Option<usize>)> {
    let mut clicked: Option<(Vec<usize>, Option<usize>)> = None;

    let is_selected = state.selected_project_path == *path_so_far;

    ui.horizontal(|ui| {
        ui.add_space(level as f32 * 16.0);
        if ui.selectable_label(is_selected, &proj.name).clicked() {
            clicked = Some((path_so_far.clone(), None));
        }
    });

    for (note_idx, note) in proj.notes.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.add_space((level + 1) as f32 * 16.0);
            if ui.button(format!("📝 {}", note.name)).clicked() {
                clicked = Some((path_so_far.clone(), Some(note_idx)));
            }
        });
    }

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
        if clicked.is_none() {
            if let Some(found) = draw_project_tree(ui, child_proj, &child_path, level + 1, state) {
                clicked = Some(found);
            }
        }
    }

    clicked
}

pub fn side_panel(ui: &mut egui::Ui, state: &mut AppState) {
    ui.heading("Projects");

    let mut maybe_clicked: Option<(Vec<usize>, Option<usize>)> = None;

    for (i, proj) in state.projects.iter().enumerate() {
        let initial_path = vec![i];
        if maybe_clicked.is_none() {
            maybe_clicked = draw_project_tree(ui, proj, &initial_path, 0, state);
        }
    }

    if let Some((new_path, maybe_note_idx)) = maybe_clicked {
        state.selected_project_path = new_path.clone();

        if let Some(note_idx) = maybe_note_idx {
            state.note_editing = Some((new_path, note_idx));
        } else {
            state.note_editing = None;
        }
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

        if ui.button("＋ Add Note").clicked() {
            if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
                proj.notes.push(Note {
                    name: "New Note".into(),
                    content: "".into(),
                });
                let note_idx = proj.notes.len() - 1;
                state.note_editing = Some((state.selected_project_path.clone(), note_idx));
            }
        }
    }

    if ui.button("＋ Add Root Project").clicked() {
        state.projects.push(Project {
            name: "New".into(),
            subprojects: Vec::new(),
            datasets: Vec::new(),
            selected_dataset: None,
            notes: Vec::new(),
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
                    notes: Vec::new(),
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
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
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
        for &tab in &[Tab::Preview, Tab::Graph, Tab::BarChart] {
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

pub fn preview_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
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
                        state.column_to_remove = None;
                        state.column_to_match = None;
                        state.match_value.clear();
                    }
                }

                if ui.button("＋ Add CSV").clicked() {
                    if let Some(path) = FileDialog::new().add_filter("CSV", &["csv"]).pick_file() {
                        let name = path
                            .file_name()
                            .and_then(|s| s.to_str())
                            .unwrap_or("csv")
                            .to_string();

                        let result: PolarsResult<DataFrame> =
                            CsvReader::from_path(&path).and_then(|r| {
                                r.has_header(true)
                                    .infer_schema(Some(10_000))
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
                            state.column_to_remove = None;
                            state.column_to_match = None;
                            state.match_value.clear();
                            ctx.request_repaint();
                        }
                    }
                }

                ui.label(format!("({} loaded)", proj.datasets.len()));
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get_mut(ds_idx) {
                    if let Some(df) = &ds.df {
                        let col_names: Vec<String> = df
                            .get_column_names()
                            .iter()
                            .map(|s| s.to_string())
                            .collect();

                        ui.horizontal(|ui| {
                            ui.label("Filter (remove one column):");

                            let selected_col = state.column_to_remove.get_or_insert_with(|| {
                                col_names.get(0).cloned().unwrap_or_default()
                            });

                            egui::ComboBox::from_id_source("remove_column_combo")
                                .selected_text(selected_col.clone())
                                .show_ui(ui, |ui| {
                                    for name in &col_names {
                                        ui.selectable_value(
                                            selected_col,
                                            name.clone(),
                                            name.clone(),
                                        );
                                    }
                                });

                            if ui.button("Remove Column").clicked() {
                                if !selected_col.is_empty() && col_names.contains(selected_col) {
                                    if let Some(orig_df) = &mut ds.df {
                                        if let Ok(mut new_df) = orig_df.drop(selected_col) {
                                            *orig_df = new_df.clone();
                                            state.column_to_remove = None;
                                            ctx.request_repaint();
                                        }
                                    }
                                }
                            }
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.label("Remove rows where");

                            let selected_match_col =
                                state.column_to_match.get_or_insert_with(|| {
                                    col_names.get(0).cloned().unwrap_or_default()
                                });

                            egui::ComboBox::from_id_source("remove_rows_combo")
                                .selected_text(selected_match_col.clone())
                                .show_ui(ui, |ui| {
                                    for name in &col_names {
                                        ui.selectable_value(
                                            selected_match_col,
                                            name.clone(),
                                            name.clone(),
                                        );
                                    }
                                });

                            ui.label("= ");

                            ui.add(
                                egui::TextEdit::singleline(&mut state.match_value)
                                    .id_source("match_value_input")
                                    .hint_text("value to match"),
                            );

                            if ui.button("Remove Rows").clicked() {
                                let input_str = state.match_value.trim().to_string();
                                if !input_str.is_empty() && col_names.contains(selected_match_col) {
                                    if let Some(orig_df) = &mut ds.df {
                                        if let Ok(s) = orig_df
                                            .column(selected_match_col)
                                            .and_then(|series| series.cast(&DataType::Utf8))
                                        {
                                            if let Ok(utf8_chunked) = s.utf8() {
                                                let equal_mask: BooleanChunked =
                                                    utf8_chunked.equal(input_str.as_str());
                                                let keep_mask = equal_mask.not();

                                                if let Ok(filtered_df) = orig_df.filter(&keep_mask)
                                                {
                                                    *orig_df = filtered_df;
                                                    state.column_to_match = None;
                                                    state.match_value.clear();
                                                    ctx.request_repaint();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        });

                        ui.separator();
                    }
                }
            }

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
                                                ui.label(format!("{:?}", val.unwrap().get_str().unwrap_or("—")));
                                            });
                                        }
                                    });
                                });
                            } else {
                                ui.label("Dataset loaded but empty.");
                            }
                        }
                    } else if proj.datasets.is_empty() {
                        ui.label("No dataset loaded.");
                    } else {
                        ui.label("Select a dataset to preview.");
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

pub fn graph_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    ui.vertical(|ui| {
        ui.heading("Graph Viewer");

        let available_size = ui.available_size();
        let (rect, response) =
            ui.allocate_exact_size(available_size, Sense::drag().union(Sense::hover()));

        if response.dragged() {
            state.graph_pan += response.drag_delta();
        }

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                let zoom_factor = (scroll * 0.005).exp();
                state.graph_zoom = (state.graph_zoom * zoom_factor).clamp(0.1, 10.0);
            }
        }

        if !state.layout_done {
            let mut positions = Vec::with_capacity(100);
            let node_spacing = 400.0;
            for i in 0..100 {
                let col = (i % 10) as f32;
                let row = (i / 10) as f32;
                positions.push(Vec2::new(col * node_spacing, row * node_spacing));
            }

            let mut edges = Vec::new();
            for i in 0..100 {
                if i % 10 < 9 {
                    edges.push((i, i + 1));
                }
                if i / 10 < 9 {
                    edges.push((i, i + 10));
                }
            }

            let n = positions.len() as f32; // = 100.0
            let area = 2000.0 * 2000.0;
            let k = (area / n).sqrt();
            let iterations = 50;
            let mut disp = vec![Vec2::ZERO; 100];

            for _ in 0..iterations {
                for d in disp.iter_mut() {
                    *d = Vec2::ZERO;
                }

                for i in 0..100 {
                    for j in (i + 1)..100 {
                        let delta = positions[i] - positions[j];
                        let dist = delta.length().max(1.0);
                        let force = (k * k) / dist;
                        let direction = delta / dist;
                        disp[i] += direction * force;
                        disp[j] -= direction * force;
                    }
                }

                for &(u, v) in &edges {
                    let delta = positions[u] - positions[v];
                    let dist = delta.length().max(1.0);
                    let force = (dist * dist) / k;
                    let direction = delta / dist;
                    disp[u] -= direction * force;
                    disp[v] += direction * force;
                }

                let temperature = 50.0;
                for i in 0..100 {
                    let d = disp[i];
                    let length = d.length().max(1.0);
                    positions[i] += (d / length) * length.min(temperature);
                }
            }

            state.node_positions = positions;
            state.layout_done = true;
        }

        let top_left = (-state.graph_pan) / state.graph_zoom;
        let world_size = rect.size() / state.graph_zoom;
        let world_rect = Rect::from_min_size(Pos2::new(top_left.x, top_left.y), world_size);
        let to_screen: RectTransform = RectTransform::from_to(world_rect, rect);
        let painter = ui.painter_at(rect);

        for i in 0..100 {
            let wp = state.node_positions[i];
            let sp = to_screen.transform_pos(Pos2::new(wp.x, wp.y));

            if i % 10 < 9 {
                let wp_r = state.node_positions[i + 1];
                let sp_r = to_screen.transform_pos(Pos2::new(wp_r.x, wp_r.y));
                painter.line_segment([sp, sp_r], (2.0, Color32::WHITE));
            }
            if i / 10 < 9 {
                let wp_d = state.node_positions[i + 10];
                let sp_d = to_screen.transform_pos(Pos2::new(wp_d.x, wp_d.y));
                painter.line_segment([sp, sp_d], (2.0, Color32::WHITE));
            }
        }

        let node_radius_world = 50.0;
        for (i, &wp) in state.node_positions.iter().enumerate() {
            let screen_pos = to_screen.transform_pos(Pos2::new(wp.x, wp.y));
            let r_screen = node_radius_world * state.graph_zoom;

            painter.circle_filled(screen_pos, r_screen, Color32::from_rgb(100, 150, 200));

            let label = format!("Node {}", i + 1);
            let text_style = egui::TextStyle::Body.resolve(ui.style());

            let offsets = [
                Vec2::new(-1.0, -1.0),
                Vec2::new(-1.0, 1.0),
                Vec2::new(1.0, -1.0),
                Vec2::new(1.0, 1.0),
            ];
            for &off in &offsets {
                let pos = screen_pos + off;
                painter.text(
                    pos,
                    Align2::CENTER_CENTER,
                    &label,
                    text_style.clone(),
                    Color32::WHITE,
                );
            }
            painter.text(
                screen_pos,
                Align2::CENTER_CENTER,
                &label,
                text_style,
                Color32::BLACK,
            );
        }
    });
}

pub fn bar_chart_tab(ctx: &egui::Context, ui: &mut Ui, state: &mut AppState) {
    if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
        ui.vertical(|ui| {
            ui.heading("[selected field] ");
            ui.heading("[selected field] entries ");

            if proj.datasets.is_empty() {
                ui.label("No datasets loaded. Go to Preview → ＋ Add CSV to load one.");
                return;
            }

            let current_idx = proj.selected_dataset.unwrap_or(0);
            let mut chosen_idx = current_idx;
            ui.horizontal(|ui| {
                ui.label("Dataset: ");
                egui::ComboBox::from_id_source("bar_dataset_combo")
                    .selected_text(
                        proj.datasets
                            .get(chosen_idx)
                            .map(|ds| ds.name.clone())
                            .unwrap_or_else(|| "<none>".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, ds) in proj.datasets.iter().enumerate() {
                            ui.selectable_value(&mut chosen_idx, i, ds.name.clone());
                        }
                    });
                if chosen_idx != current_idx {
                    proj.selected_dataset = Some(chosen_idx);
                    state.bar_genes.clear();
                    state.bar_counts.clear();
                }
            });

            ui.separator();

            if let Some(ds_idx) = proj.selected_dataset {
                if let Some(ds) = proj.datasets.get(ds_idx) {
                    let mut df = match CsvReader::from_path(&ds.path)
                        .and_then(|rdr| rdr
                            .infer_schema(None)
                            .has_header(true)
                            .finish())
                    {
                        Ok(df) => df,
                        Err(err) => {
                            ui.colored_label(egui::Color32::RED, format!("CSV load error: {}", err));
                            return;
                        }
                    };

                    if state.bar_genes.is_empty() && !state.is_fetching {
                        state.is_fetching = true;
                        let key = state.bar_selected_field.clone();
                        let value = state.bar_selected_value.clone();
                        let mut df_for_request = df.clone();

                        let response_result: Result<String, reqwest::Error> = request_bar_chart_blocking(&mut df_for_request, key, value);

                        state.is_fetching = false;

                        match response_result {
                            Ok(text) => {
                                match serde_json::from_str::<Value>(&text) {
                                    Ok(json) => {
                                        if let (Some(genes_arr), Some(counts_arr)) = (
                                            json.get("genes").and_then(|v| v.as_array()),
                                            json.get("counts").and_then(|v| v.as_array()),
                                        ) {
                                            state.bar_genes = genes_arr
                                                .iter()
                                                .map(|v| v.as_str().unwrap_or("").to_string())
                                                .collect();

                                            state.bar_counts = counts_arr
                                                .iter()
                                                .map(|v| v.as_f64().unwrap_or(0.0))
                                                .collect();
                                        } else {
                                            ui.colored_label(
                                                egui::Color32::RED,
                                                "Unexpected JSON format: missing 'genes' or 'counts'",
                                            );
                                            return;
                                        }
                                    }
                                    Err(err) => {
                                        ui.colored_label(
                                            egui::Color32::RED,
                                            format!("JSON parse error: {}", err),
                                        );
                                        return;
                                    }
                                }
                            }
                            Err(err) => {
                                ui.colored_label(
                                    egui::Color32::RED,
                                    format!("Request error: {}", err),
                                );
                                return;
                            }
                        }
                    }

                    if !state.bar_genes.is_empty() && state.bar_genes.len() == state.bar_counts.len()
                    {
                        let genes = &state.bar_genes;
                        let counts = &state.bar_counts;

                        let bars: Vec<Bar> = counts
                            .iter()
                            .enumerate()
                            .map(|(i, &c)| Bar::new(i as f64, c))
                            .collect();

                        Plot::new("real_diseases_per_gene")
                            .x_axis_formatter(|grid_mark, _range| {
                                let idx = grid_mark.value.round() as usize;
                                if idx < genes.len() {
                                    genes[idx].clone()
                                } else {
                                    String::new()
                                }
                            })
                            .show(ui, |plot_ui| {
                                plot_ui.bar_chart(BarChart::new(bars));
                            });
                    } else if state.is_fetching {
                        ui.label("Loading chart…");
                    } else {
                        ui.label("No chart data available. Select a dataset above.");
                    }
                } else {
                    ui.label("Invalid dataset index.");
                }
            } else {
                ui.label("Select a dataset from the dropdown above.");
            }
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.label("Select or create a project first.");
        });
    }
}

// pub fn plot_tab(ui: &mut egui::Ui, state: &mut AppState) {
//     if let Some(proj) = get_project_mut(&mut state.projects, &state.selected_project_path) {
//         ui.vertical(|ui| {
//             if let Some(ds_idx) = proj.selected_dataset {
//                 if let Some(ds) = proj.datasets.get(ds_idx) {
//                     if let Some(df) = &ds.df {
//                         let mut grouped: DataFrame =
//                             df.groupby(["GeneSymbol"]).unwrap().count().unwrap();

//                         let first_count_col: String = grouped
//                             .get_column_names()
//                             .iter()
//                             .find(|name| name.ends_with("_count"))
//                             .expect("Expected at least one column ending in \"_count\"")
//                             .to_string();

//                         grouped.rename(&first_count_col, "count").unwrap();

//                         let counts_series: Series = grouped
//                             .column("count")
//                             .unwrap()
//                             .cast(&DataType::Int64)
//                             .unwrap()
//                             .clone();

//                         let counts_chunked = counts_series.i64().unwrap();
//                         let count_vec: Vec<i64> = counts_chunked.into_no_null_iter().collect();

//                         let mut freq_map: HashMap<i64, i64> = HashMap::new();
//                         for &num_diseases in &count_vec {
//                             *freq_map.entry(num_diseases).or_insert(0) += 1;
//                         }

//                         let mut bins: Vec<(f64, f64)> = freq_map
//                             .into_iter()
//                             .map(|(num_diseases, num_genes)| {
//                                 (num_diseases as f64, num_genes as f64)
//                             })
//                             .collect();
//                         bins.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

//                         let bars: Vec<Bar> =
//                             bins.into_iter().map(|(x, _)| Bar::new(x, 0.8)).collect();

//                         Plot::new("diseases_per_gene_histogram")
//                             .view_aspect(2.0)
//                             .show(ui, |plot_ui| {
//                                 plot_ui.bar_chart(
//                                     BarChart::new(bars).color(Color32::from_rgb(100, 150, 250)),
//                                 );
//                             });
//                     } else {
//                         ui.label("Dataset loaded but empty.");
//                     }
//                 } else {
//                     ui.label("Invalid dataset index.");
//                 }
//             } else {
//                 ui.label("Select a dataset to plot.");
//             }
//         });
//     } else {
//         // No project selected at all
//         ui.centered_and_justified(|ui| {
//             ui.label("Select or create a project first.");
//         });
//     }
// }
