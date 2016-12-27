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
enum Data {
    RGBA(u32, u32, u32, u32),
    Empty,
}

impl Default for Data {
    fn default() -> Data {
        Data::Empty
    }
}

#[derive(Debug, Copy, Clone, Default)]
struct Point {
    x: i64,
    y: i64,
}

/// A bounding box. `min` is the upper left point, `max` is the lower right point.
#[derive(Debug)]
struct Bounds {
    min: Point,
    max: Point,
}

impl Bounds {
    /// Determine the size of the bounding box.
    pub fn dimensions(&self) -> (i64, i64) {
        let width = self.max.x - self.min.x;
        let height = self.max.y - self.min.y;
        (width, height)
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

#[derive(Debug)]
struct Grid {
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
    pub fn translate(&mut self, x: i64, y: i64) {
        let mut new_plane = HashMap::new();
        for (point, data) in self.plane.iter() {
            let x_new = point.0 + x;
            let y_new = point.1 + y;
            new_plane.insert((x_new, y_new), *data);
        }
        self.plane = new_plane;
    }

    /// Does the gruntwork for flipping the `Grid`.
    fn do_flip(&mut self, points: Vec<Point>) {
        let mut new_plane: HashMap<(i64, i64), Data> = HashMap::new();

        // Need a reverse and forward iterator for interchanging of point coordinates.
        let mut points_reversed = points.iter().rev();
        let mut points = points.iter();

        let mut i = 0;
        let max_iterations = self.plane.len();

        while i < max_iterations {
            // Get the source point coordinates.
            if let Some(src) = points.next() {
                // Get the target point coordinates.
                if let Some(target) = points_reversed.next() {
                    // Get the data from the source point.
                    if let Some(data) = self.get(src.x, src.y) {
                        // Add the source data to the target point coordinates.
                        new_plane.insert((target.x, target.y), *data);
                    }
                }
            }
            i += 1;
        }
        self.plane = new_plane;
    }

    /// Flip the entire `Grid` horizontally.
    pub fn flip_horizontally(&mut self) {
        let mut points =
            self.plane.keys().map(|p| Point { x: p.0, y: p.1 }).collect::<Vec<Point>>();

        // Want to sort by x-coordinate since this is a horizontal flip.
        points.sort_by_key(|p| p.x);

        self.do_flip(points);
    }

    /// Flip the entire `Grid` vertically..
    pub fn flip_vertically(&mut self) {
        let mut points =
            self.plane.keys().map(|p| Point { x: p.0, y: p.1 }).collect::<Vec<Point>>();

        // Want to sort by y-coordinate since this is a vertical flip.
        points.sort_by_key(|p| p.y);

        self.do_flip(points);
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
            assert_eq!(dimensions.0, 1);
            assert_eq!(dimensions.1, 2);
        }

        it "translates coordinates" {
            let mut grid = grid;
            let data1 = Data::RGBA(1,1,1,1);
            let data2 = Data::RGBA(2,2,2,2);
            grid.set(1,1,data1);
            grid.set(2,3,data2);
            grid.translate(1, -1);
            let point1 = grid.get(2, 0).unwrap();
            let point2 = grid.get(3, 2).unwrap();
            assert_eq!(*point1, Data::RGBA(1,1,1,1));
            assert_eq!(*point2, Data::RGBA(2,2,2,2));
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
            assert_eq!(dimensions.0, 2);
            assert_eq!(dimensions.1, 1);
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
