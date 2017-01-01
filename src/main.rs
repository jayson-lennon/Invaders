extern crate invaderlib;
use invaderlib::{ImageTool, Grid};
use invaderlib::Gen::Generator;
use ImageTool::{new_file, save_grid, GridExporter};

fn main() {
    let mut gen = Generator::new();
    let mut num_generated = 0;
    let mut sheet = Grid::new();
    let mut rows = 0;
    let num_per_row = 8;
    let mut failed_gens = 0;
    while num_generated < 64 {
        if failed_gens >= 1000 {
            println!("Maximum number of failed generations reached. Stopping generation.");
            break;
        }
        if let Some(invader) = gen.invader(8, 8, 5, 12, (1, 3), (1, 5), 1) {
            let width = invader.size().0;
            sheet.merge_at(&invader,
                           (num_generated % num_per_row) * 8 * 2 + (16 / width),
                           rows * 8 * 2);
            if (num_generated + 1) % num_per_row == 0 && num_generated > 0 {
                rows += 1;
            }
        } else {
            failed_gens += 1;
            println!("Failed to generate an invader. Retry #{}", failed_gens);
            continue;
        }
        num_generated += 1;
    }
    if let Ok(fh) = new_file("invader.png") {
        if save_grid(fh,
                     &sheet,
                     (num_generated / num_per_row * 8 * 2) as u32,
                     (num_generated / num_per_row * 8 * 2) as u32)
            .is_err() {
            println!("Failed composing image data");
        }
    }

}
