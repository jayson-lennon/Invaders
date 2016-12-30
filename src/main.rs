extern crate invaderlib;
use invaderlib::{ImageTool, Grid};
use invaderlib::Gen::Generator;
use ImageTool::{new_file, save_grid, GridExporter};

fn main() {
    let mut gen = Generator::new();
    let invader = gen.invader(8, 16, (7, 7), (8, 14), (1, 1), (1, 5));
    if let Ok(fh) = new_file("invader.png") {
        if save_grid(fh, &invader, 16, 16).is_err() {
            println!("Failed composing image data");
        }
    }
}
