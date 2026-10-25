// include/strata/platform/nvme_cache.hpp - Async NVMe Storage Engine for Tiered KV Cache.
// Provides non-blocking Direct I/O (O_DIRECT / FILE_FLAG_NO_BUFFERING), worker queue,
// crash-consistent manifest management, and LRU eviction policy.
#pragma once

#include "strata/core/kv_block.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace strata::platform {

using strata::core::AlignedBuffer;
using strata::core::KvBlock;
using strata::core::KvBlockHeader;
using strata::core::kKvBlockAlignment;

/// Configuration for NvmeCache storage engine.
struct NvmeCacheConfig {
    std::filesystem::path cache_dir = ".strata_kv_cache";
    size_t disk_budget_bytes = 10ULL * 1024 * 1024 * 1024; // 10 GiB default budget
    size_t num_worker_threads = 4;
    bool direct_io = true;                  // Attempt Direct I/O (O_DIRECT on Linux, FILE_FLAG_NO_BUFFERING on Win)
    bool auto_direct_io_fallback = true;   // Automatically fallback to buffered I/O if FS rejects Direct I/O
    bool sync_manifest_on_write = false;   // Atomically flush manifest after each write operation
};

/// Cache runtime statistics.
struct NvmeCacheStats {
    uint64_t total_bytes = 0;       // Total disk bytes used by blocks
    uint64_t block_count = 0;       // Total blocks in cache
    uint64_t pinned_count = 0;      // Number of pinned blocks
    uint64_t read_hits = 0;         // Successful block reads
    uint64_t read_misses = 0;       // Failed or missing block reads
    uint64_t write_count = 0;       // Successful block writes
    uint64_t write_errors = 0;      // Failed block writes
    uint64_t evictions = 0;         // Number of evicted blocks
    uint64_t bytes_evicted = 0;     // Total bytes evicted
};

/// Manifest header stored at the beginning of .strata_kv_manifest.bin (64 bytes).
#pragma pack(push, 1)
struct ManifestHeader {
    char magic[8] = {'S', 'T', 'R', 'M', 'A', 'N', 'I', 'F'}; // "STRMANIF"
    uint32_t version = 1;
    uint32_t record_count = 0;
    uint64_t total_bytes = 0;
    uint64_t checksum = 0;          // Checksum of all records data
    uint64_t reserved[4] = {0};
};
#pragma pack(pop)
static_assert(sizeof(ManifestHeader) == 64, "ManifestHeader must be exactly 64 bytes");

/// Record stored for each persisted KV block in the manifest (48 bytes).
#pragma pack(push, 1)
struct ManifestBlockRecord {
    uint64_t block_hash = 0;
    uint64_t file_size = 0;
    uint64_t last_access_us = 0;
    uint64_t parent_hash = 0;
    uint32_t num_tokens = 0;
    uint32_t is_pinned = 0;         // 1 if pinned, 0 if unpinned
    int32_t format = 0;
    uint32_t reserved = 0;

    bool operator==(const ManifestBlockRecord& o) const = default;
};
#pragma pack(pop)
static_assert(sizeof(ManifestBlockRecord) == 48, "ManifestBlockRecord must be exactly 48 bytes");

// Callback signatures
using WriteCallback = std::function<void(bool ok, const std::string& err)>;
using ReadBlockCallback = std::function<void(bool ok, KvBlock block, const std::string& err)>;
using ReadBufferCallback = std::function<void(bool ok, AlignedBuffer buffer, const std::string& err)>;

/// Non-blocking NVMe storage engine for KV blocks.
class NvmeCache {
public:
    NvmeCache();
    explicit NvmeCache(NvmeCacheConfig config);
    explicit NvmeCache(const std::filesystem::path& cache_dir);
    ~NvmeCache();

    NvmeCache(const NvmeCache&) = delete;
    NvmeCache& operator=(const NvmeCache&) = delete;
    NvmeCache(NvmeCache&&) = delete;
    NvmeCache& operator=(NvmeCache&&) = delete;

    // ---- Asynchronous Operations ----

    /// Asynchronously write a serialized or deserialized KvBlock.
    void write_block_async(const KvBlock& block, WriteCallback callback = nullptr);
    void write_block_async(const KvBlock& block, bool is_pinned, WriteCallback callback = nullptr);

    /// Asynchronously write raw 4096-aligned buffer.
    void write_buffer_async(uint64_t block_hash,
                            AlignedBuffer buffer,
                            uint32_t num_tokens = 0,
                            bool is_pinned = false,
                            WriteCallback callback = nullptr);

    /// Asynchronously read a block and deserialize it into a KvBlock.
    void read_block_async(uint64_t block_hash, ReadBlockCallback callback);

    /// Asynchronously read raw aligned block bytes without deserialization.
    void read_buffer_async(uint64_t block_hash, ReadBufferCallback callback);

    // ---- Synchronous Wrappers ----

    bool write_block_sync(const KvBlock& block, std::string* err = nullptr);
    bool write_block_sync(const KvBlock& block, bool is_pinned, std::string* err = nullptr);
    bool write_buffer_sync(uint64_t block_hash,
                           const AlignedBuffer& buffer,
                           uint32_t num_tokens = 0,
                           bool is_pinned = false,
                           std::string* err = nullptr);

    bool read_block_sync(uint64_t block_hash, KvBlock& out_block, std::string* err = nullptr);
    bool read_buffer_sync(uint64_t block_hash, AlignedBuffer& out_buffer, std::string* err = nullptr);

    /// Drain all queued tasks and write manifest atomically to disk.
    void flush_sync();

    // ---- Pinning & Eviction Management ----

    /// Pin or unpin a block to protect it from LRU eviction.
    bool pin_block(uint64_t block_hash, bool pinned = true);
    bool unpin_block(uint64_t block_hash);
    bool is_pinned(uint64_t block_hash) const;

    /// Check if block is present in cache.
    bool contains(uint64_t block_hash) const;

    /// Retrieve block metadata from manifest index.
    std::optional<ManifestBlockRecord> get_metadata(uint64_t block_hash) const;

    /// Evict oldest unpinned blocks to fit additional bytes within disk_budget_bytes.
    /// Returns the number of blocks evicted.
    size_t evict_to_budget(size_t additional_bytes = 0);

    /// Manually remove a block from cache and disk.
    bool remove_block(uint64_t block_hash);

    /// Clear all cached blocks and reset manifest.
    void clear();

    // ---- Paths, Stats & Direct I/O state ----

    std::filesystem::path get_block_path(uint64_t block_hash) const;
    std::filesystem::path manifest_path() const;
    NvmeCacheStats get_stats() const;
    void reset_stats();
    const NvmeCacheConfig& config() const { return config_; }
    bool is_direct_io_active() const;

private:
    struct Task {
        enum class Type { kWrite, kReadBlock, kReadBuffer } type;
        uint64_t block_hash = 0;
        std::shared_ptr<const AlignedBuffer> write_buffer;
        uint32_t num_tokens = 0;
        bool is_pinned = false;
        uint64_t parent_hash = 0;
        int32_t format = 0;

        WriteCallback write_cb;
        ReadBlockCallback read_block_cb;
        ReadBufferCallback read_buffer_cb;
    };

    struct InFlightEntry {
        std::shared_ptr<const AlignedBuffer> buffer;
        uint32_t num_tokens = 0;
        bool is_pinned = false;
        uint64_t parent_hash = 0;
        int32_t format = 0;
    };

    void init_directories();
    void load_manifest();
    bool save_manifest_atomic();
    bool save_manifest_atomic_locked();
    void worker_loop();
    void process_task(Task& task);

    bool write_file_direct(const std::filesystem::path& path, const uint8_t* data, size_t size, std::string& err);
    bool read_file_direct(const std::filesystem::path& path, AlignedBuffer& out_buf, size_t expected_size, std::string& err);

    size_t evict_to_budget_locked(size_t additional_bytes);

    NvmeCacheConfig config_;
    mutable std::mutex manifest_mu_;
    std::unordered_map<uint64_t, ManifestBlockRecord> manifest_index_;
    std::unordered_map<uint64_t, InFlightEntry> in_flight_writes_;
    NvmeCacheStats stats_;
    bool manifest_dirty_ = false;
    mutable std::atomic<bool> direct_io_supported_ = true;

    // Worker queue
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
    std::condition_variable drain_cv_;
    std::deque<Task> task_queue_;
    size_t active_workers_ = 0;
    bool stop_workers_ = false;
    std::vector<std::thread> workers_;
};

} // namespace strata::platform
