// Copyright 2016 Jayson Lennon
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#![feature(plugin)]
#![cfg_attr(test, plugin(stainless))]

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate image;

use std::fs::File;
use std::path::Path;
use std::collections::HashMap;

use image::{GenericImage, Pixel, ImageBuffer};

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Data {
    RGBA(u8, u8, u8, u8),
    Empty,
}

impl Default for Data {
    fn default() -> Data {
        Data::Empty
    }
}
/// Used internally for determining which direction to flip a Grid.
enum Flip {
    Horizontally,
    Vertically,
}

#[derive(Debug, Copy, Clone, Default)]
pub struct Point {
    x: i64,
    y: i64,
}

/// A bounding box. `min` is the lower left point, `max` is the upper right point.
#[derive(Debug)]
pub struct Bounds {
    min: Point,
    max: Point,
}

pub trait OutputPx {
    fn get_pixel_at(&self, x: u32, y: u32) -> Data;
}
impl Bounds {
    /// Determine the size of the bounding box.
    pub fn dimensions(&self) -> (i64, i64) {
        let width = self.max.x - self.min.x;
        let height = self.max.y - self.min.y;
        (width + 1, height + 1)
    }

    /// Create new `Bound`s from a collection of `Point`s.
    pub fn from_points<'a, T>(points: T) -> Bounds
        where T: Iterator<Item = &'a Point>
    {
        let mut xmin = 0;
        let mut ymin = 0;
        let mut xmax = 0;
        let mut ymax = 0;
        let mut initialized = false;

        for point in points {
            if initialized {
                // Handle x-coordinates.
                if point.x < xmin {
                    xmin = point.x
                }
                if point.x > xmax {
                    xmax = point.x
                }
                // Handle y-coordinates.
                if point.y < ymin {
                    ymin = point.y
                }
                if point.y > ymax {
                    ymax = point.y
                }
            } else {
                // Set up initial values.
                xmin = point.x;
                ymin = point.y;
                xmax = point.x;
                ymax = point.y;
                initialized = true;
            }
        }

        let min = Point { x: xmin, y: ymin };
        let max = Point { x: xmax, y: ymax };

        Bounds {
            min: min,
            max: max,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Grid {
    plane: HashMap<(i64, i64), Data>,
}

impl Grid {
    pub fn new() -> Grid {
        Grid { plane: HashMap::new() }
    }

    /// Gets coordinate `Data` at (x,y). Returns None if coordinate does not exist, otherwise
    /// returns a reference to `Data`.
    pub fn get(&self, x: i64, y: i64) -> Option<&Data> {
        self.plane.get(&(x, y))
    }

    /// Sets coordinate (x,y) to `Data`. Returns None if the coordinate added is a new coordinate.
    /// Returns the old `Data` from the coordinate if the coordinate was previously added.
    pub fn set(&mut self, x: i64, y: i64, data: Data) -> Option<Data> {
        self.plane.insert((x, y), data)
    }

    /// Calculates a bounding box containing the pixels of the `Grid`.
    pub fn bounds(&self) -> Bounds {
        let points = self.plane.keys().map(|p| Point { x: p.0, y: p.1 }).collect::<Vec<Point>>();
        Bounds::from_points(points.iter())
    }

    /// Calculates the size of the bounding box containing the pixels of the `Grid`.
    pub fn size(&self) -> (i64, i64) {
        self.bounds().dimensions()
    }

    /// Translates the `Point`s in the `Grid` by (x,y).
    pub fn translate_by(&mut self, x: i64, y: i64) {
        let mut new_plane = HashMap::new();
        for (point, data) in &self.plane {
            let x_new = point.0 + x;
            let y_new = point.1 + y;
            new_plane.insert((x_new, y_new), *data);
        }
        self.plane = new_plane;
    }

    /// Translates the `Point`s in the `Grid` to (x,y). The origin is the lower left coordinate
    /// of a bounding box encompassing all `Point`s.
    pub fn translate_to(&mut self, x: i64, y: i64) {
        let bounds = self.bounds();
        let tx = x - bounds.min.x;
        let ty = y - bounds.min.y;
        self.translate_by(tx, ty);
    }

    /// Merge another `Grid` into this `Grid` at the provided (x,y) coordinates. The origin of the
    /// `Grid` being added is the lower left corner of a bounding box encompassing all `Point`s
    /// in the `Grid`.
    pub fn merge_at(&mut self, grid: &Grid, x: i64, y: i64) {
        let mut translated_grid = grid.clone();
        translated_grid.translate_to(x, y);
        self.merge(&translated_grid);
    }

    /// Merge another `Grid` into this `Grid`.
    pub fn merge(&mut self, grid: &Grid) {
        for (point, data) in grid.plane.iter() {
            self.plane.insert(*point, *data);
        }
    }

    /// Removes all elements from the `Grid`.
    pub fn clear(&mut self) {
        self.plane = HashMap::new();
    }

    /// Replace the current `Grid` with new_grid.
    pub fn replace_all(&mut self, new_grid: Grid) {
        self.plane = new_grid.plane;
    }

    /// Does the gruntwork for flipping the `Grid`
    fn do_flip(&mut self, flip: Flip) {
        let mut new_plane: HashMap<(i64, i64), Data> = HashMap::new();
        // Iterate through all points and apply the approriate translation.
        for (coords, data) in self.plane.iter() {
            match flip {
                Flip::Horizontally => {
                    new_plane.insert((-coords.0, coords.1), *data);
                }
                Flip::Vertically => {
                    new_plane.insert((coords.0, -coords.1), *data);
                }
            };
        }
        self.plane = new_plane;
    }

    /// Flip the entire `Grid` horizontally. Pivot is about the y-axis.
    /// Quadrant I & IV will be flipped to Quadrant II & III, respectively.
    pub fn flip_horizontally(&mut self) {
        self.do_flip(Flip::Horizontally);
    }

    /// Flip the entire `Grid` vertically. Pivot is about the x-axis.
    /// Quadrant I & II will be flipped to Quadrant III & IV, respectively.
    pub fn flip_vertically(&mut self) {
        self.do_flip(Flip::Vertically);
    }
}

impl OutputPx for Grid {
    fn get_pixel_at(&self, x: u32, y: u32) -> Data {
        match self.get(x as i64, y as i64) {
            Some(data) => return *data,
            None => return Data::RGBA(0, 0, 0, 255),
        }
    }
}

/// Helper functions for working with cartesian coordinates.
mod CellTool {

    use super::Grid;
    /// Determines the how many "steps" must be taken to reach the target coordinates from the
    /// source coordinates.
    /// Ex: (1,1) -> (0,0) = (-1,-1)
    pub fn offset(source: (i64, i64), target: (i64, i64)) -> (i64, i64) {
        let (xs, ys) = (source.0, source.1);
        let (xt, yt) = (target.0, target.1);

        let x_offset = xt - xs;
        let y_offset = yt - ys;

        (x_offset, y_offset)
    }

    /// Determines the distance between the source and target coordinates.
    pub fn distance(source: (i64, i64), target: (i64, i64)) -> f32 {
        let (xs, ys) = (source.0 as f32, source.1 as f32);
        let (xt, yt) = (target.0 as f32, target.1 as f32);

        ((xt - xs).powi(2) + (yt - ys).powi(2)).sqrt()
    }

    /// Determines if the target coordinates are adjacent to the source coordinates.
    pub fn is_adjacent(source: (i64, i64), target: (i64, i64)) -> bool {
        let (xs, ys) = (source.0, source.1);
        let (xt, yt) = (target.0, target.1);

        let x_distance = (xs - xt).abs();
        let y_distance = (ys - yt).abs();

        // The coordinates cannot be more than 1 space away.
        if x_distance > 1 || y_distance > 1 {
            return false;
        }

        // For a cell to be adjacent, only the x or y coordinate may be 1 space away.
        // If both are 1 space away, then it is a corner cell.
        if x_distance == 1 && y_distance == 1 {
            return false;
        }

        return true;
    }

    /// Gets all coordinates that are directly adjacent to the target coordinates.
    pub fn get_adjacent_coords(target: (i64, i64)) -> Vec<(i64, i64)> {
        let mut coords: Vec<(i64, i64)> = Vec::new();
        coords.push((target.0 - 1, target.1));
        coords.push((target.0 + 1, target.1));
        coords.push((target.0, target.1 - 1));
        coords.push((target.0, target.1 + 1));
        coords
    }

    /// Gets all coordinates that are corners to the target coordinates.
    pub fn get_corner_coords(target: (i64, i64)) -> Vec<(i64, i64)> {
        let mut coords: Vec<(i64, i64)> = Vec::new();
        coords.push((target.0 - 1, target.1 - 1));
        coords.push((target.0 - 1, target.1 + 1));
        coords.push((target.0 + 1, target.1 - 1));
        coords.push((target.0 + 1, target.1 + 1));
        coords
    }

    /// Gets all coordinates that are surrounding the target coordinates.
    pub fn get_surrounding_coords(target: (i64, i64)) -> Vec<(i64, i64)> {
        let mut surrounding = Vec::new();
        let mut corners = get_corner_coords(target);
        let mut adjacent = get_adjacent_coords(target);
        surrounding.append(&mut corners);
        surrounding.append(&mut adjacent);
        surrounding
    }

    #[cfg(test)]
    mod tests {
        use super::{get_adjacent_coords, is_adjacent, get_surrounding_coords, get_corner_coords,
                    offset, distance, expand_cells};
        use super::super::{Grid, Data, Point, Bounds};
        describe! celltool {

            it "determines if a cell is adjacent" {
                let source = (1,1);
                let adjacent = (1,2);
                assert_eq!(is_adjacent(source, adjacent), true);
            }

            it "determines if a non-adjacent cell is non-adjacent" {
                let source = (1,1);
                let nonadjacent = (2,2);
                assert_eq!(is_adjacent(source, nonadjacent), false);
            }

            it "gets adjacent cell coordinates" {
                let target = (0,0);
                let coords = vec![(-1,0),(1,0),(0,-1),(0,1)];
                let adjacent = get_adjacent_coords(target);
                assert_eq!(adjacent, coords);
            }

            it "gets corner cell coordinates" {
                let target = (0,0);
                let coords = vec![(-1,-1),(-1,1),(1,-1),(1,1)];
                let corners = get_corner_coords(target);
                assert_eq!(corners, coords);
            }

            it "gets surrounding cell coordinates" {
                let target = (0,0);
                /// Surrounding is corners + adjacent, so copy-pasta those tests in that order.
                let coords = vec![(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1)];
                let surrounding = get_surrounding_coords(target);
                assert_eq!(surrounding, coords);
            }

            it "calculates negative offsets" {
                let source = (0,0);
                let target = (-2,-2);
                let offset = offset(source, target);
                assert_eq!(offset.0, -2);
                assert_eq!(offset.1, -2);
            }

            it "calculates positive offsets" {
                let source = (0,0);
                let target = (2,2);
                let offset = offset(source, target);
                assert_eq!(offset.0, 2);
                assert_eq!(offset.1, 2);
            }

            it "calculates distance" {
                let source = (0,0);
                let target = (3,2);
                let d = distance(source, target);
                assert_eq!(d, 13_f32.sqrt());
            }
        }
    }

}

mod GridTool {
    use super::{Grid, Data};

    #[cfg(test)]
    mod tests {
        use super::{Grid, Data};

        describe! gridtool {
            before_each {
                let grid = Grid::new();
            }
            it "does nothing" {
                
            }
        }

    }
}

pub mod ImageTool {
    use std::fs::File;
    use std::io;
    use std::path::Path;
    use std::collections::HashMap;

    extern crate image;
    use image::{GenericImage, Pixel, ImageBuffer, Primitive};

    use super::{OutputPx, Data, Point, Grid};

    pub trait GridExporter {
        fn from_grid<T>(pixgrid: &T,
                        src_size: (u32, u32))
                        -> ImageBuffer<image::Rgba<u8>, Vec<u8>>
            where T: OutputPx;
    }

    impl GridExporter for ImageBuffer<image::Rgba<u8>, Vec<u8>> {
        fn from_grid<T>(pixgrid: &T, src_size: (u32, u32)) -> ImageBuffer<image::Rgba<u8>, Vec<u8>>
            where T: OutputPx
        {
            let (imgx, imgy) = src_size;
            // Create a new ImgBuf with width: imgx and height: imgy
            let mut imgbuf = image::ImageBuffer::new(imgx, imgy);
            let mut x = 0;
            let mut y = 0;
            // Iterate through all pixels.
            loop {
                match pixgrid.get_pixel_at(x, y) {
                    Data::RGBA(r, g, b, a) => {
                        let px = image::Rgba([r, g, b, a]);
                        // Transform y-coordinate from y-axis up to y-axis down.
                        let y_coord = imgy - y - 1;
                        imgbuf.put_pixel(x as u32, y_coord as u32, px);
                    }
                    _ => (),
                }
                x += 1;
                // Check to see if we hit the rightmost x coordinate.
                if x == imgx {
                    // If we did, start x-coordinate at leftmost for next row.
                    x = 0;
                    // Increment y-coordinate to start next row,
                    y += 1;
                    // Bail out once we hit the maximum y-coordinate.
                    if y == imgy {
                        break;
                    }
                }
            }
            imgbuf
        }
    }

    pub fn new_file(path: &str) -> Result<File, io::Error> {
        File::create(&Path::new(path))
    }

    pub fn save_png(mut fh: File,
                    imgbuf: ImageBuffer<image::Rgba<u8>, Vec<u8>>)
                    -> Result<(), image::ImageError> {
        let ref mut fh = fh;
        image::ImageRgba8(imgbuf).save(fh, image::PNG)
    }

    pub fn save_grid(mut fh: File,
                     grid: &Grid,
                     width: u32,
                     height: u32)
                     -> Result<(), image::ImageError> {
        let ref mut fh = fh;
        let buf = ImageBuffer::from_grid(grid, (width, height));
        image::ImageRgba8(buf).save(fh, image::PNG)
    }

    #[cfg(test)]
    mod tests {
        use ImageTool;
        use image::ImageBuffer;
        use super::GridExporter;
        use super::super::{Grid, Data, Point, Bounds};

        describe! imagetool {
            before_each {
                let grid = Grid::new();
            }

            // Grid pixels are stored y-axis up. ImageBuffer pixels are stored y-axis down.
            // This test ensures that the conversion from grid to image coordinates are correctly
            // mapped so the image isn't exported upside-down.
            it "converts a Grid to an image buffer" {
                let mut grid = grid;
                let white = Data::RGBA(255,255,255,255);
                let red = Data::RGBA(255,0,0,255);
                grid.set(0,0,red);
                grid.set(1,1,white);
                grid.set(2,2,red);
                let size = grid.bounds().dimensions();
                let buf = ImageBuffer::from_grid(&grid, (size.0 as u32, size.1 as u32));
                let pix_00 = buf.get_pixel(0,2);
                let pix_11 = buf.get_pixel(1,1);
                let pix_22 = buf.get_pixel(2,0);
                assert_eq!(pix_00.data[0], 255);
                assert_eq!(pix_00.data[1], 0);
                assert_eq!(pix_00.data[2], 0);

                assert_eq!(pix_11.data[0], 255);
                assert_eq!(pix_11.data[1], 255);
                assert_eq!(pix_11.data[2], 255);

                assert_eq!(pix_22.data[0], 255);
                assert_eq!(pix_22.data[1], 0);
                assert_eq!(pix_22.data[2], 0);
            }
        }
    }
}

/// Generates pixel art.
pub mod Gen {
    extern crate rand;
    use Gen::rand::{Rng, SeedableRng, StdRng, ThreadRng, thread_rng};
    use Gen::rand::distributions::{Range, IndependentSample};
    use super::{Grid, Data, CellTool};

    /// Container for random number generators.
    pub struct Generator {
        seed_gen: ThreadRng,
        rng: StdRng,
    }

    impl Generator {
        /// Create new random number generators.
        pub fn new() -> Generator {
            let mut seed_gen = thread_rng();
            let seed: &[_] = &[seed_gen.gen(), seed_gen.gen(), seed_gen.gen()];
            let rng: StdRng = StdRng::from_seed(seed);
            Generator {
                seed_gen: seed_gen,
                rng: rng,
            }
        }

        /// Generate a new seed to be used by the RNG.
        pub fn new_seed<'a>(&'a mut self) -> Vec<usize> {
            vec![self.seed_gen.gen(), self.seed_gen.gen(), self.seed_gen.gen()]
        }

        /// Generate and automatically reseed the RNG.
        pub fn reseed(&mut self) {
            let seed = self.new_seed();
            self.rng.reseed(seed.as_slice());
        }

        /// Create an invader with a pre-supplied seed.
        /// seed:
        pub fn invader_seeded(&mut self,
                              seed: Vec<usize>,
                              width: u32,
                              height: u32,
                              min_px: u32,
                              max_px: u32,
                              max_nearby: (u32, u32),
                              max_edge: (u32, u32),
                              center_overlap: i32)
                              -> Option<Grid> {
            // Set up RNG seed to the supplied seed.
            self.rng.reseed(seed.as_slice());

            let mut max_px = max_px;
            if min_px >= max_px {
                max_px = min_px + 1;
            }
            // Pick values from a range based on supplied params.
            let total_pixels = Range::new(min_px, max_px + 1).ind_sample(&mut self.rng);

            // Maximum pixels that are allowed to be "nearby" (corner and adjacent pixels).
            let max_nearby = Range::new(max_nearby.0, max_nearby.1 + 1).ind_sample(&mut self.rng);
            // Maxmimum number of pixels that may be along the right edge.
            let max_edge = Range::new(max_edge.0, max_edge.1 + 1).ind_sample(&mut self.rng);

            // Create ranges for the x and y coordinates based on desired width and height.
            let x_rng = Range::new(0, width);
            let y_rng = Range::new(0, height + 1);

            // This is the Grid where we will be writing pixels.
            let mut grid = Grid::new();

            // Total pixels filled by generation.
            let mut pixels_filled = 0;

            // Number of pixels that are on the right edge of the image.
            let mut edge_pixels = 0;

            println!("px: {}:{}->{}, nearby: {}, edge: {}",
                     min_px,
                     max_px,
                     total_pixels,
                     max_nearby,
                     max_edge);

            // Bailout iteration counter in case the generation loops too many times.
            // Hardcoded to limit @ 10000.
            let mut iterations = 0;

            while pixels_filled < total_pixels {
                iterations += 1;
                // Bailout if we try to too many times. Supplied parameters are causing problems
                // with the algorithm and the params should be adjusted.
                if iterations > 10000 {
                    return None;
                }
                // Pick the x and y coordinates.
                let x = x_rng.ind_sample(&mut self.rng);
                let y = y_rng.ind_sample(&mut self.rng);

                // Don't reuse the same pixel twice.
                if grid.get(x as i64, y as i64).is_some() {
                    continue;
                }

                // If we have an edge pixel, check if we are already at the limit.
                if x == width - 1 && edge_pixels >= max_edge {
                    continue;
                }
                // The pixel picked is an edge pixel.
                if x == width - 1 {
                    edge_pixels += 1;
                }

                // Get all "nearby" pixels (adjacent and corners). This is so we can adjust
                // how "clumped up" the pixels get. In most cases, we won't want to have
                // a bunch of pixels making a giant rectangle.
                let nearby_pixels = CellTool::get_surrounding_coords((x as i64, y as i64));
                // Number of pixels found that are nearby.
                let mut num_nearby = 0;
                for coords in nearby_pixels {
                    // Check for used pixels nearby.
                    if grid.get(coords.0 as i64, coords.1 as i64).is_some() {
                        num_nearby += 1;
                    }
                }

                // If this is our first pixel, then we will be using it. If not, we need to make
                // sure that it is "nearby" at least 1 other pixel but not exceeding the maximum
                // "nearby" pixels as defined by max_nearby.
                if pixels_filled == 0 || (num_nearby > 0 && num_nearby <= max_nearby) {
                    // We are just filling them with white for now. Need to add color/palette
                    // options at some point in the future.
                    let pixel = Data::RGBA(255, 255, 255, 255);
                    // If this is the first pixel then we are going to set it at the right edge
                    // since we always require at least 1 pixel on the right edge.
                    if pixels_filled == 0 {
                        grid.set((width - 1) as i64, y as i64, pixel);
                    } else {
                        // Pixels afterwards can be anywhere (subject to above constraints).
                        grid.set(x as i64, y as i64, pixel);
                    }
                    pixels_filled += 1;
                }
            }

            // Duplicating the grid so we can mirror the image.
            let mut dup_grid = grid.clone();
            // Flip the grid, placing the coordinates into Quadrant II.
            dup_grid.flip_horizontally();
            // We translate by width*2: once to bring the grid back to Quadrant I, and again to
            // bring it next to the original grid to create a mirror image. -1 is for zero-indexing.
            // center_overlap determines how the image is mirrored:
            // == 0: image will be mirrored
            //  < 0: image will be disconnected from original by center_overlap pixels.
            //  > 0: image will intersect the original by center_overlap pixels
            dup_grid.translate_by((width as i64 * 2) - 1 - center_overlap as i64, 0);
            grid.merge(&dup_grid);

            // Check if we are working with a non-mirror image and if so, we need to center it.
            if center_overlap != 0 {
                // Note that odd values for either image size or center_overlap will always result in
                // an off-center image.
                grid.translate_by((-center_overlap / 2) as i64, 0);
            }
            Some(grid)
        }

        /// Create an invader with a randomly generated seed.
        pub fn invader(&mut self,
                       width: u32,
                       height: u32,
                       min_px: u32,
                       max_px: u32,
                       max_nearby: (u32, u32),
                       max_edge: (u32, u32),
                       center_overlap: i32)
                       -> Option<Grid> {
            let seed = self.new_seed();
            println!("Gen invader with seed :{:?}", seed);
            self.invader_seeded(seed,
                                width,
                                height,
                                min_px,
                                max_px,
                                max_nearby,
                                max_edge,
                                center_overlap)
        }
    }

    #[cfg(test)]
    mod tests {
        describe! generate {
            it "need generator tests" {
                assert!(false, "need tests to ensure basic generation");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Grid, Data, Point, Bounds};

    describe! grid {
        before_each {
            let grid = Grid::new();
        }
        
        it "adds and reads data" {
            let mut grid = grid;
            let data = Data::RGBA(1,1,1,1);
            grid.set(1,2,data);
            let read_data = grid.get(1,2).unwrap();
            assert_eq!(*read_data, Data::RGBA(1,1,1,1));
        }

        it "properly handles getting invalid data" {
            let read_data = grid.get(9,9);
            assert_eq!(read_data.is_some(), false);
        }

        it "reports proper bounding box" {
            let mut grid = grid;
            let data1: Data = Default::default();
            let data2: Data = Default::default();
            grid.set(1,1,data1);
            grid.set(2,2,data2);
            let bounds = grid.bounds();
            assert_eq!(bounds.min.x, 1);
            assert_eq!(bounds.min.y, 1);
            assert_eq!(bounds.max.x, 2);
            assert_eq!(bounds.max.y, 2);
        }

        it "reports proper dimensions" {
            let mut grid = grid;
            let data1: Data = Default::default();
            let data2: Data = Default::default();
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            let dimensions = grid.size();
            assert_eq!(dimensions.0, 2);
            assert_eq!(dimensions.1, 3);
        }

        it "translates by coordinates" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            grid.translate_by(1, -1);
            let point1 = grid.get(2, 0).unwrap();
            let point2 = grid.get(3, 2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

        it "translates to coordinates" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            grid.translate_to(0, 0);
            let point1 = grid.get(0, 0).unwrap();
            let point2 = grid.get(1, 2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

        it "horizontally flips the grid" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            let data3 = Data::RGBA(4,4,4,4);
            grid.set(-1,-1,data1);
            grid.set(3,2,data2);
            grid.set(4,3,data3);
            grid.flip_horizontally();
            let point1 = grid.get(1, -1).unwrap();
            let point2 = grid.get(-3, 2).unwrap();
            let point3 = grid.get(-4, 3).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
            assert_eq!(*point3, data3);
        }

        it "vertically flips the grid" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            let data3 = Data::RGBA(4,4,4,4);
            grid.set(-1,-1,data1);
            grid.set(3,2,data2);
            grid.set(4,3,data3);
            grid.flip_vertically();
            let point1 = grid.get(-1, 1).unwrap();
            let point2 = grid.get(3, -2).unwrap();
            let point3 = grid.get(4, -3).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
            assert_eq!(*point3, data3);
        }

        it "merges another grid" {
            let mut grid = grid;
            let mut add_grid = Grid::new();
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);

            add_grid.set(1, 1, data1);
            grid.set(2,2, data2);

            grid.merge(&add_grid);

            let point1 = grid.get(1,1).unwrap();
            let point2 = grid.get(2,2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

        it "merges another grid at a coordinate" {
            let mut grid = grid;
            let mut add_grid = Grid::new();
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);

            add_grid.set(1, 1, data1);
            grid.set(2,2, data2);

            grid.merge_at(&add_grid, 0, 0);

            let point1 = grid.get(0,0).unwrap();
            let point2 = grid.get(2,2).unwrap();
            assert_eq!(*point1, data1);
            assert_eq!(*point2, data2);
        }

    }

    describe! bounds {
        before_each {
            let grid = Grid::new();
        }

        it "calculates dimensions" {
            let min = Point { x:0, y:0 };
            let max = Point { x:2, y:1 };
            let bounds = Bounds { min: min, max: max };
            let dimensions = bounds.dimensions();
            assert_eq!(dimensions.0, 3);
            assert_eq!(dimensions.1, 2);
        }

        it "determines bounding points from a collection" {
            let min = Point { x:0, y:0 };
            let max = Point { x:2, y:1 };
            let points = vec![max, min];
            let bounds = Bounds::from_points(points.iter());
            assert_eq!(bounds.min.x, 0);
            assert_eq!(bounds.min.y, 0);
            assert_eq!(bounds.max.x, 2);
            assert_eq!(bounds.max.y, 1);
        }
    }
}
