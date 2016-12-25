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

#![feature(conservative_impl_trait)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_snake_case)]
extern crate image;

use std::fs::File;
use std::path::Path;

use image::{GenericImage, Pixel, ImageBuffer};

#[derive(Debug, PartialEq)]
enum Data {
    RGBA(u32, u32, u32, u32),
    Empty,
}

#[derive(Debug)]
struct Point {
    x: usize,
    y: usize,
}

#[derive(Debug)]
struct Grid {
    plane: Vec<Data>,
    dimensions: Point,
}

impl Grid {
    fn new(x: usize, y: usize) -> Grid {
        let mut plane = Vec::new();
        plane.reserve_exact(x * y);

        let mut i = 0;
        while i < (x * y) {
            plane.push(Data::Empty);
            i += 1;
        }

        Grid {
            plane: plane,
            dimensions: Point { x: x, y: y },
        }
    }

    fn get(&self, x: usize, y: usize) -> Option<&Data> {
        self.plane.get((x * self.dimensions.x) + y)
    }


    fn set(&mut self, x: usize, y: usize, data: Data) -> bool {
        if let Some(cell) = self.plane.get_mut((x * self.dimensions.x) + y) {
            *cell = data;
            return true;
        } else {
            return false;
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
                let grid = Grid::new(3,3);
            }
            it "does nothing" {
                
            }
        }

    }
}



#[cfg(test)]
mod tests {
    use super::{Grid, Data};

    describe! grid {
        before_each {
            let grid = Grid::new(3,3);
        }
        
        it "adds and reads data" {
            let mut grid = grid;
            let data = Data::RGBA(1,1,1,1);
            grid.set(1,2,data);
            let read_data = grid.get(1,2).unwrap();
            assert_eq!(*read_data, Data::RGBA(1,1,1,1));
        }

        it "properly handles out of range get operation" {
            let read_data = grid.get(9,9);
            assert_eq!(read_data.is_some(), false);
        }

        it "properly handles out of range set operation" {
            let mut grid = grid;
            let data = Data::RGBA(1,1,1,1);
            let op = grid.set(9,9,data);
            assert_eq!(op, false);
        }
    }
}
