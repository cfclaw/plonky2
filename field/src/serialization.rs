#[cfg(feature = "serialize_speedy")]
use speedy::{Readable, Writable, LittleEndian};

// We define a simpler, cleaner MaybeSpeedy.
// By making the HRTB a supertrait, we "leak" the proof to the compiler
// so it can satisfy the derive macro's requirements.
#[cfg(all(feature = "serialize_speedy", target_endian = "little"))]
pub trait MaybeSpeedy: 
    for<'a> Readable<'a, LittleEndian> + 
    Writable<LittleEndian> 
{}

#[cfg(all(feature = "serialize_speedy", target_endian = "little"))]
impl<T> MaybeSpeedy for T 
where 
    T: for<'a> Readable<'a, LittleEndian> + Writable<LittleEndian> 
{}

#[cfg(not(all(feature = "serialize_speedy", target_endian = "little")))]
pub trait MaybeSpeedy {}

#[cfg(not(all(feature = "serialize_speedy", target_endian = "little")))]
impl<T> MaybeSpeedy for T {}


#[cfg(all(feature="serialize_bytemuck", target_endian = "little"))]
pub trait MaybeBytemuck: Sized + bytemuck::Pod + bytemuck::Zeroable {}
#[cfg(all(feature="serialize_bytemuck", target_endian = "little"))]
impl<T: Sized + bytemuck::Pod + bytemuck::Zeroable> MaybeBytemuck for T {}
#[cfg(not(all(feature="serialize_bytemuck", target_endian = "little")))]
pub trait MaybeBytemuck {}
#[cfg(not(all(feature="serialize_bytemuck", target_endian = "little")))]
impl<T> MaybeBytemuck for T {}

#[cfg(feature = "ts-rs")]
pub trait MaybeTSRS: ts_rs::TS {}
#[cfg(feature = "ts-rs")]
impl<T: ts_rs::TS> MaybeTSRS for T {}
#[cfg(not(feature = "ts-rs"))]
pub trait MaybeTSRS {}
#[cfg(not(feature = "ts-rs"))]
impl<T> MaybeTSRS for T {}

pub trait MaybePsySerialize: MaybeSpeedy + MaybeBytemuck {}
impl<T: MaybeSpeedy + MaybeBytemuck> MaybePsySerialize for T {}