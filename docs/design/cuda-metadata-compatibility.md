# CUDA metadata compatibility

`CUDAMetadata::cuda_query_weights()` and the four `CUDAColumnData` missing-flag
device getters retain their signatures and values. Their first call now uploads
an owned host snapshot to the object's GPU and completes the transfer before
returning. Repeated calls return the same pointer until the corresponding
metadata is replaced or the object is destroyed. First access and snapshot
replacement are serialized; callers must finish using returned pointers before
metadata mutation or destruction. Lazy allocation and release restore the
calling thread's previously selected device.

Query snapshots are replaced by `Init` and `SetQuery`; a null query-weight
pointer clears old weights. `SetQuery(nullptr, nullptr, 0)` clears boundaries
as well. Column snapshots are replaced on initialization, view creation and
every subset refresh. They own their inputs, including when inputs are temporary
or a view borrows column buffers. Normal column-buffer view lifetime rules still
apply.

These snapshots add one host copy of query weights and four bytes per feature
for flags. This keeps the compatibility API independent of parent lifetimes
while avoiding five device allocations/uploads during normal training, which
does not call these getters. The first getter pays for allocation and a device
completion fence. The explicit fence at metadata initialization remains.

`CUDAMetadata::SetInitScore` remains callable as a no-op: its former device
mirror had no getter or consumer. Host `Metadata` initial scores, the score
updater and RF's separate device score buffer retain their behavior.
`CUDARowData`'s old host-mapped accessors still return false and an empty pointer.

Device metadata owns resources and cannot be copied or moved. The previous
implicit `CUDAMetadata` copy shallow-copied owning device pointers and was unsafe;
column sharing continues through `InitViewOf`. Installed C++ class layouts have
changed and require a source rebuild.
