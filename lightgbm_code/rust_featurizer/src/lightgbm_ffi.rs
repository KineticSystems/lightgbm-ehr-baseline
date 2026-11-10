//! Minimal dynamic bindings for LightGBM's public C API.
//!
//! The library is loaded from an explicit path so the executor provably runs
//! the exact `lib_lightgbm.so` installed in the Python environment; both
//! executors therefore share one set of machine code and must produce
//! byte-identical models. Only the entry points the trial executor needs are
//! bound, and every call goes through `check`, which surfaces
//! `LGBM_GetLastError` on failure.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int, c_void};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;

pub(crate) type DatasetHandle = *mut c_void;
pub(crate) type BoosterHandle = *mut c_void;

macro_rules! bind {
    ($lib:ident, $name:ident, $type:ty) => {{
        let symbol = CString::new(stringify!($name)).unwrap();
        let pointer = unsafe { dlsym($lib, symbol.as_ptr()) };
        if pointer.is_null() {
            bail!("lib_lightgbm is missing symbol {}", stringify!($name));
        }
        unsafe { std::mem::transmute::<*mut c_void, $type>(pointer) }
    }};
}

type FnGetLastError = unsafe extern "C" fn() -> *const c_char;
type FnGetSampleCount =
    unsafe extern "C" fn(c_int, *const c_char, *mut c_int) -> c_int;
type FnSampleIndices =
    unsafe extern "C" fn(c_int, *const c_char, *mut c_void, *mut c_int) -> c_int;
type FnDatasetCreateFromSampledColumn = unsafe extern "C" fn(
    *mut *mut c_double,
    *mut *mut c_int,
    c_int,
    *const c_int,
    c_int,
    c_int,
    i64,
    *const c_char,
    *mut DatasetHandle,
) -> c_int;
type FnDatasetCreateByReference =
    unsafe extern "C" fn(DatasetHandle, i64, *mut DatasetHandle) -> c_int;
type FnDatasetInitStreaming = unsafe extern "C" fn(
    DatasetHandle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type FnDatasetPushRowsWithMetadata = unsafe extern "C" fn(
    DatasetHandle,
    *const c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_float,
    *const c_float,
    *const c_double,
    *const c_int,
    c_int,
) -> c_int;
type FnDatasetMarkFinished = unsafe extern "C" fn(DatasetHandle) -> c_int;
type FnDatasetSetFeatureNames =
    unsafe extern "C" fn(DatasetHandle, *const *const c_char, c_int) -> c_int;
type FnDatasetFree = unsafe extern "C" fn(DatasetHandle) -> c_int;
type FnBoosterCreate =
    unsafe extern "C" fn(DatasetHandle, *const c_char, *mut BoosterHandle) -> c_int;
type FnBoosterAddValidData =
    unsafe extern "C" fn(BoosterHandle, DatasetHandle) -> c_int;
type FnBoosterUpdateOneIter =
    unsafe extern "C" fn(BoosterHandle, *mut c_int) -> c_int;
type FnBoosterGetEval = unsafe extern "C" fn(
    BoosterHandle,
    c_int,
    *mut c_int,
    *mut c_double,
) -> c_int;
type FnBoosterGetEvalCounts =
    unsafe extern "C" fn(BoosterHandle, *mut c_int) -> c_int;
type FnBoosterGetCurrentIteration =
    unsafe extern "C" fn(BoosterHandle, *mut c_int) -> c_int;
type FnBoosterRollbackOneIter = unsafe extern "C" fn(BoosterHandle) -> c_int;
type FnBoosterGetNumPredict =
    unsafe extern "C" fn(BoosterHandle, c_int, *mut i64) -> c_int;
type FnBoosterGetPredict =
    unsafe extern "C" fn(BoosterHandle, c_int, *mut i64, *mut c_double) -> c_int;
type FnBoosterSaveModelToString = unsafe extern "C" fn(
    BoosterHandle,
    c_int,
    c_int,
    c_int,
    i64,
    *mut i64,
    *mut c_char,
) -> c_int;
type FnBoosterLoadModelFromString =
    unsafe extern "C" fn(*const c_char, *mut c_int, *mut BoosterHandle) -> c_int;
type FnBoosterPredictForMat = unsafe extern "C" fn(
    BoosterHandle,
    *const c_void,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_char,
    *mut i64,
    *mut c_double,
) -> c_int;
type FnBoosterFree = unsafe extern "C" fn(BoosterHandle) -> c_int;

pub(crate) const C_API_DTYPE_FLOAT32: c_int = 0;
pub(crate) const C_API_PREDICT_NORMAL: c_int = 0;

pub(crate) struct LightGbm {
    get_last_error: FnGetLastError,
    pub(crate) get_sample_count: FnGetSampleCount,
    pub(crate) sample_indices: FnSampleIndices,
    pub(crate) dataset_create_from_sampled_column: FnDatasetCreateFromSampledColumn,
    pub(crate) dataset_create_by_reference: FnDatasetCreateByReference,
    pub(crate) dataset_init_streaming: FnDatasetInitStreaming,
    pub(crate) dataset_push_rows_with_metadata: FnDatasetPushRowsWithMetadata,
    pub(crate) dataset_mark_finished: FnDatasetMarkFinished,
    pub(crate) dataset_set_feature_names: FnDatasetSetFeatureNames,
    pub(crate) dataset_free: FnDatasetFree,
    pub(crate) booster_create: FnBoosterCreate,
    pub(crate) booster_add_valid_data: FnBoosterAddValidData,
    pub(crate) booster_update_one_iter: FnBoosterUpdateOneIter,
    pub(crate) booster_get_eval: FnBoosterGetEval,
    pub(crate) booster_get_eval_counts: FnBoosterGetEvalCounts,
    pub(crate) booster_get_current_iteration: FnBoosterGetCurrentIteration,
    pub(crate) booster_rollback_one_iter: FnBoosterRollbackOneIter,
    pub(crate) booster_get_num_predict: FnBoosterGetNumPredict,
    pub(crate) booster_get_predict: FnBoosterGetPredict,
    pub(crate) booster_save_model_to_string: FnBoosterSaveModelToString,
    pub(crate) booster_load_model_from_string: FnBoosterLoadModelFromString,
    pub(crate) booster_predict_for_mat: FnBoosterPredictForMat,
    pub(crate) booster_free: FnBoosterFree,
}

// Function pointers into a library we never unload; the handles the executor
// shares across trial threads are const-safe per the thread-safety audit.
unsafe impl Send for LightGbm {}
unsafe impl Sync for LightGbm {}

impl LightGbm {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let raw_path = CString::new(path.to_str().context("non-UTF-8 library path")?)
            .context("library path contains a NUL byte")?;
        let library = unsafe { dlopen(raw_path.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if library.is_null() {
            let detail = unsafe {
                let message = dlerror();
                if message.is_null() {
                    "unknown dlopen failure".to_string()
                } else {
                    CStr::from_ptr(message).to_string_lossy().into_owned()
                }
            };
            bail!("cannot dlopen {}: {detail}", path.display());
        }
        Ok(Self {
            get_last_error: bind!(library, LGBM_GetLastError, FnGetLastError),
            get_sample_count: bind!(library, LGBM_GetSampleCount, FnGetSampleCount),
            sample_indices: bind!(library, LGBM_SampleIndices, FnSampleIndices),
            dataset_create_from_sampled_column: bind!(
                library,
                LGBM_DatasetCreateFromSampledColumn,
                FnDatasetCreateFromSampledColumn
            ),
            dataset_create_by_reference: bind!(
                library,
                LGBM_DatasetCreateByReference,
                FnDatasetCreateByReference
            ),
            dataset_init_streaming: bind!(
                library,
                LGBM_DatasetInitStreaming,
                FnDatasetInitStreaming
            ),
            dataset_push_rows_with_metadata: bind!(
                library,
                LGBM_DatasetPushRowsWithMetadata,
                FnDatasetPushRowsWithMetadata
            ),
            dataset_mark_finished: bind!(
                library,
                LGBM_DatasetMarkFinished,
                FnDatasetMarkFinished
            ),
            dataset_set_feature_names: bind!(
                library,
                LGBM_DatasetSetFeatureNames,
                FnDatasetSetFeatureNames
            ),
            dataset_free: bind!(library, LGBM_DatasetFree, FnDatasetFree),
            booster_create: bind!(library, LGBM_BoosterCreate, FnBoosterCreate),
            booster_add_valid_data: bind!(
                library,
                LGBM_BoosterAddValidData,
                FnBoosterAddValidData
            ),
            booster_update_one_iter: bind!(
                library,
                LGBM_BoosterUpdateOneIter,
                FnBoosterUpdateOneIter
            ),
            booster_get_eval: bind!(library, LGBM_BoosterGetEval, FnBoosterGetEval),
            booster_get_eval_counts: bind!(
                library,
                LGBM_BoosterGetEvalCounts,
                FnBoosterGetEvalCounts
            ),
            booster_get_current_iteration: bind!(
                library,
                LGBM_BoosterGetCurrentIteration,
                FnBoosterGetCurrentIteration
            ),
            booster_rollback_one_iter: bind!(
                library,
                LGBM_BoosterRollbackOneIter,
                FnBoosterRollbackOneIter
            ),
            booster_get_num_predict: bind!(
                library,
                LGBM_BoosterGetNumPredict,
                FnBoosterGetNumPredict
            ),
            booster_get_predict: bind!(
                library,
                LGBM_BoosterGetPredict,
                FnBoosterGetPredict
            ),
            booster_save_model_to_string: bind!(
                library,
                LGBM_BoosterSaveModelToString,
                FnBoosterSaveModelToString
            ),
            booster_load_model_from_string: bind!(
                library,
                LGBM_BoosterLoadModelFromString,
                FnBoosterLoadModelFromString
            ),
            booster_predict_for_mat: bind!(
                library,
                LGBM_BoosterPredictForMat,
                FnBoosterPredictForMat
            ),
            booster_free: bind!(library, LGBM_BoosterFree, FnBoosterFree),
        })
    }

    pub(crate) fn check(&self, code: c_int, operation: &str) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        let message = unsafe {
            let raw = (self.get_last_error)();
            if raw.is_null() {
                "unknown LightGBM error".to_string()
            } else {
                CStr::from_ptr(raw).to_string_lossy().into_owned()
            }
        };
        Err(anyhow!("{operation} failed: {message}"))
    }
}
