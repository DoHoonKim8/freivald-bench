//! This crate provides various common gadgets and chips for use with `halo2_proofs`.
//!
//! # Gadgets
//!
//! Gadgets are an abstraction for writing reusable and interoperable circuit logic. They
//! do not create any circuit constraints or assignments themselves, instead interacting
//! with the circuit through a defined "instruction set". A circuit developer uses gadgets
//! by instantiating them with a particular choice of chip.
//!
//! # Chips
//!
//! Chips implement the low-level circuit constraints. The same instructions may be
//! implemented by multiple chips, enabling different performance trade-offs to be made.
//! Chips can be highly optimised by their developers, as long as they conform to the
//! defined instructions.

#![cfg_attr(docsrs, feature(doc_cfg))]
// Temporary until we have more of the crate implemented.
#![allow(dead_code, clippy::single_range_in_vec_init)]
// Catch documentation errors caused by code changes.
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod mat_mul;
pub mod poseidon;
pub mod utilities;
