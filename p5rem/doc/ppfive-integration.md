---
updated: ["2026-04-15T00:00"]
---

# PPFive Integration Investigation

## Executive Summary

**Feasibility: HIGH.** ppfive has a nearly API-compatible interface with pyfive. Integration is **very achievable** with ~200-300 lines of additional code. The main differences are in chunking support and some missing indexing utilities, but these can be worked around.

## API Compatibility Analysis

### ✅ Directly Compatible

| Feature | pyfive | ppfive | Status |
|---------|--------|--------|--------|
| File open | `pyfive.File(path)` | `ppfive.File(path)` | ✅ Same API |
| Variable access | `file[varname]` | `file[varname]` | ✅ Same API |
| Variable attributes | `var.attrs` | `var.attrs` | ✅ Same API |
| Variable shape/dtype | `var.shape`, `var.dtype` | `var.shape`, `var.dtype` | ✅ Same API |
| File attributes | `file.attrs` | `file.attrs` | ✅ Same API |
| File keys | `file.keys()` | `file.keys()` | ✅ Same API |
| Dataset ID | `dataset.id` | `dataset.id` | ✅ Same API (ppfive.variable.VariableID) |
| Chunk index | `id.index` | `id.index` | ✅ Same dict structure (chunk_coords → StoreInfo) |
| Raw chunk read | `id._get_raw_chunk(storeinfo)` | `id.read_direct_chunk(chunk_coords)` | ⚠️ Similar but slightly different |

### ⚠️ Partially Compatible

| Feature | pyfive | ppfive | Impact |
|---------|--------|--------|--------|
| Get chunk by storeinfo | Returns raw bytes directly | Requires chunk coordinates | LOW — wrapper needed |
| get_chunk_info_by_coord | ✅ | ✅ (same name) | ✅ No change needed |
| Chunk structure | `StoreInfo(byte_offset, size, filter_mask)` | `StoreInfo(chunk_offset, byte_offset, size, filter_mask)` | LOW — both work, extra field ignored |
| Contiguous layout | `data_offset` on dataset.id | Not documented, may not exist | ⚠️ Need to check |

### ❌ Missing in PPFive

| Feature | Required by p5rem? | Workaround |
|---------|------------------|-----------|
| `OrthogonalIndexer` | YES (chunk-worklist parallel reduce) | Must rewrite chunk planning, or use ppfive's `id.iter_chunks()` + materialization for now |
| `ZarrArrayStub` | YES (chunk planning) | Not needed if using `id.iter_chunks()` |
| `_decode_chunk()` | OPTIONAL (decompression) | ppfive.Variable exposes `_data_loader` and `_materialize()`, different architecture |

## Current pyfive Usage in remote_server.py

### File Open (Line 208)
```python
import pyfive
file_handle = pyfive.File(path)
```

### Parallel Reduction with Chunk Planning (Lines 564-571)
```python
from pyfive.h5d import ZarrArrayStub
from pyfive.indexing import OrthogonalIndexer
indexer = OrthogonalIndexer(indexer_args, ZarrArrayStub(shape, tuple(chunks)))
work_items = list(dataset_id._get_required_chunks(indexer))
```

### Chunk Decode (Lines 579-589)
```python
decode_chunk = getattr(dataset_id, "_decode_chunk", None)
chunk_array = decode_chunk(raw, filter_mask, dtype)
```

## Integration Strategy

### Approach A: Format-Aware Backend Selection (RECOMMENDED)
**Effort: 1-2 days | Risk: Low | User impact: None**

Detect file format first, then open with appropriate backend:

```python
def handle_file_open(self, path: str) -> dict[str, Any]:
    file_format = self._detect_file_format(path)
    
    if file_format == "hdf5":
        import pyfive
        file_handle = pyfive.File(path)
    elif file_format == "pp":
        try:
            import ppfive
            file_handle = ppfive.File(path)
        except ImportError:
            raise RuntimeError(
                f"ppfive not installed. Cannot open PP file {path}. "
                f"Install ppfive with: pip install ppfive"
            )
    else:
        raise ValueError(f"Unknown file format: {path}")
    
    self._open_files[path] = file_handle
    self._file_formats[path] = file_format
    return {...}

def _detect_file_format(self, path: str) -> str:
    """Detect file format from extension and magic bytes."""
    path_lower = str(path).lower()
    
    # Extension first
    if path_lower.endswith(('.h5', '.hdf5', '.nc', '.netcdf')):
        return 'hdf5'
    if path_lower.endswith(('.pp', '.pp.gz', '.fields')):
        return 'pp'
    
    # Magic bytes fallback
    with open(path, 'rb') as f:
        magic = f.read(4)
    
    if magic == b'\x89HDF':  # HDF5 signature
        return 'hdf5'
    if magic[:2] in (b'\x00\x00', b'\x00\x01'):  # Possible UM PP
        return 'pp'
    
    raise ValueError(f"Unknown file format: {path}")
```

**Backend-specific handling**:
- **HDF5 (pyfive)**: Full parallel reduction with chunk-worklist (existing code)
- **PP (ppfive)**: Use `var._materialize()` for reductions (acceptable for small PP files)

**Advantages**:
- No ambiguity about which backend to use
- Clear error messages when ppfive is missing
- Each backend handles its own format optimally

### Approach B: Abstracted File Adapter
**Effort: 3-4 days | Risk: Low | User impact: None**

Wrap format detection and backend selection in an adapter layer:

```python
class AbstractFile:
    """Wraps pyfive.File or ppfive.File based on detected format."""
    
    def __init__(self, path):
        fmt = self._detect_format(path)
        if fmt == "hdf5":
            import pyfive
            self._file = pyfive.File(path)
        elif fmt == "pp":
            import ppfive
            self._file = ppfive.File(path)
        else:
            raise ValueError(f"Unknown format: {path}")
        self._backend = fmt
    
    def __getitem__(self, key):
        return AbstractVariable(self._file[key], self._backend)
    
    def _detect_format(self, path: str) -> str:
        """Detect HDF5 vs PP from extension/magic bytes."""
        # Extension check, then magic bytes (as above)
        ...

class AbstractVariable:
    """Wraps pyfive.Variable or ppfive.Variable with consistent interface."""
    
    def __init__(self, var, backend):
        self._var = var
        self._backend = backend
    
    # For HDF5: direct pyfive methods
    # For PP: wrap ppfive methods (possibly materializing data)
```

**Benefits**: Format detection centralized, backend differences hidden in adapter

### Approach C: Backend-Specific Branches
**Effort: 2-3 days | Risk: Medium | User impact: None**

Keep existing pyfive code, add conditional branches for ppfive:

```python
def _parallel_reduce_selection(self, ...):
    backend = self._file_backends.get(path, "pyfive")
    
    if backend == "pyfive":
        # Existing OrthogonalIndexer path
        ...
    elif backend == "ppfive":
        # ppfive-specific path using iter_chunks()
        ...
```

**Pros**: Minimal refactoring, explicit code paths
**Cons**: Code duplication, harder to maintain

## Recommended Implementation Plan

### Phase 1 (Immediate): Approach A with Format Detection
**Timeline: ~4 hours | Impact: Enables full PP support with reasonable limitations**

1. Implement `_detect_file_format()` in ServerStub
2. Update `handle_file_open()` to:
   - Detect format (HDF5 vs PP)
   - Open with pyfive (HDF5) or ppfive (PP)
   - Store format flag for later use
3. Update reduction methods to check format:
   - If pyfive/HDF5: use existing chunk-worklist code
   - If ppfive/PP: use materialization or simple iteration
4. Add clear error messages when ppfive missing for PP files
5. Test with both HDF5 and PP files

### Phase 2 (Later): Approach B for Code Cleanliness
**Timeline: ~3-4 days | Impact: Cleaner architecture, easier maintenance**

- Build adapter layer once code stabilizes
- Hides format differences behind consistent interface
- Makes adding new formats easier in future

## Format Detection Approach

Use file extension + magic bytes (shown in Approach A code above):

**Extension checks (in order)**:
- `.h5`, `.hdf5`, `.nc`, `.netcdf` → HDF5
- `.pp`, `.pp.gz`, `.fields` → PP

**Magic byte fallback**:
- `\x89HDF` (4 bytes) → HDF5
- `\x00\x00` or `\x00\x01` (first 2 bytes) → Possible UM PP

**Error handling**:
- Unknown format → raise `ValueError`
- Format detected but backend missing → raise `RuntimeError` with installation instructions

## Error Handling Strategy

**Format detection failures**:
- Unknown extension and unrecognizable magic bytes → `ValueError("Unknown file format: ...")`
- Clear message helps users identify file corruption or unsupported formats

**Backend failures**:
- HDF5 file but pyfive missing → Should not happen (pyfive is required dependency)
- PP file but ppfive missing → `RuntimeError("ppfive not installed. Install with: pip install ppfive")`
- File corruption or permission issues → Let underlying library errors propagate with context

**Example error flow**:
```
user opens /data/file.pp
  ↓
detect_file_format() → "pp"
  ↓
try: import ppfive ✗
  ↓
RuntimeError: "ppfive not installed. Cannot open PP file /data/file.pp. 
             Install ppfive with: pip install ppfive"
```

## Testing Strategy

1. **Test data**:
   - Keep existing HDF5 test files in `tests/data/` (e.g., `test1.nc`, `contiguous_eg.nc`)
   - Add small PP test file (if available or can be created with ppfive)

2. **Format detection tests**:
   - `test_detect_file_format_hdf5()` — `.h5`, `.hdf5`, `.nc` files
   - `test_detect_file_format_pp()` — `.pp`, `.fields` files
   - `test_detect_file_format_by_magic_bytes()` — magic byte detection
   - `test_detect_file_format_unknown()` — error on unknown format

3. **Backend-specific tests**:
   - `test_loopback_hdf5_file_open()` — existing pyfive path
   - `test_loopback_pp_file_open()` — ppfive path (skip if ppfive not installed)
   - `test_loopback_pp_var_access()` — ppfive variable indexing
   - `test_loopback_pp_reduce_selection()` — ppfive reduction (materialized)

4. **Dependency tests**:
   - `test_pp_file_without_ppfive_raises()` — clear error when ppfive missing
   - Mark ppfive tests with `@pytest.mark.skipif(ppfive_not_available)`

5. **CI**:
   - Default: run with pyfive only (HDF5 tests pass, PP tests skipped)
   - Optional: run with ppfive to test both paths

## Summary Decision Table

| Aspect | Feasibility | Effort | Risk |
|--------|-------------|--------|------|
| Format detection | ✅ Trivial | <30 min | None |
| ppfive File open | ✅ Trivial | <30 min | None |
| ppfive var access | ✅ Trivial | <30 min | None |
| ppfive full-dataset reduce | ✅ 1 hour | Low (materialize) | Low |
| ppfive partial-selection reduce | ✅ 1 hour | Low (materialize subset) | Low |
| ppfive single-chunk reduce | ✅ 1-2 hours | Medium (chunk lookup) | Low |
| ppfive parallel reduce | ⚠️ Limited | 4-6 hours | High (needs rewrite) |
| Error handling & testing | ✅ Easy | 1-2 hours | Low |

## Conclusion

**Recommendation**: Implement Phase 1 (Format-aware backend selection) → **~4 hours of work**, enables full PP file support with reasonable limitations.

Format-aware approach provides clear value:
- **No ambiguity**: File format determines backend upfront
- **Transparent support**: Users with ppfive installed can open PP files seamlessly
- **Clear failures**: Users without ppfive get helpful error messages with installation guidance
- **Unchanged HDF5 path**: Existing pyfive (HDF5) code and optimizations remain untouched
- **Deferred optimization**: Parallel reduction for PP files deferred to Phase 2 if performance becomes a bottleneck

**Key insight**: Each backend handles only its own format, so we don't need fallback logic—just smart upfront detection.
