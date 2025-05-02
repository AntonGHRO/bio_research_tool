use eframe::{egui, App, Frame, NativeOptions, run_native};
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Deserialize)]
struct Message {
    message: String,
}

struct AppState {
    last_message: String,
    is_fetching: bool,
}

struct MyApp {
    client: Client,
    state: Arc<Mutex<AppState>>,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            client: Client::new(),
            state: Arc::new(Mutex::new(AppState {
                last_message: "Press the button…".into(),
                is_fetching: false,
            })),
        }
    }
}

impl App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut st = self.state.lock().unwrap();

            if st.is_fetching {
                ui.label("Fetching…");
            } else if ui.button("Fetch from Python").clicked() {
                st.is_fetching = true;
                let client = self.client.clone();
                let state = Arc::clone(&self.state);
                thread::spawn(move || {
                    let resp = client
                        .get("http://127.0.0.1:8000/hello")
                        .send()
                        .and_then(|r| r.json::<Message>());
                    let mut st = state.lock().unwrap();
                    match resp {
                        Ok(msg) => st.last_message = msg.message,
                        Err(err) => st.last_message = format!("Error: {}", err),
                    }
                    st.is_fetching = false;
                });
            }

            ui.separator();
            ui.label(&st.last_message);
        });

        // continuously repaint to catch state updates from the background thread
        ctx.request_repaint();
    }
}

fn main() {
    let native_options = NativeOptions::default();
    run_native(
        "egui ↔ FastAPI Demo",
        native_options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}
